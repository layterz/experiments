import random
import numpy as np
import torch
import einops
import pandas as pd
from fancy_einsum import einsum
from rich import print
from transformer_lens import HookedTransformer
from plotly import express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

pio.renderers.default = "png"
plt.ioff()

def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device
    
def get_head_index(i):
    return (i // 12, i % 12)

def get_subgroup_label(layer_index, head_index):
    single_value = (layer_index * 12) + head_index
    return chr(single_value + ord('A'))

def run_prompts(model: HookedTransformer, *prompts, **kwargs):
    device = get_device()
    _, cache = model.run_with_cache(list(prompts), **kwargs)
    cache.prompts = list(prompts)
    cache.device = device
    return cache

def decompose_head(cache, l, h, pos=None):
    start, end = 0, len(cache['q', l][0])
    if pos is not None:
        start, end = pos
    q = cache['q', l][:, start:end, h, :]
    k = cache['k', l][:, start:end, h, :]
    v = cache['v', l][:, start:end, h, :]
    return q, k, v

def project_attn(model, l, h, c):
    OV = model.OV[l, h]
    proj_A = torch.einsum('b n m h, d h -> b n m d', c, OV.A)
    proj_B = torch.einsum('b n m d, h d -> b n m d', proj_A, OV.B)
    return proj_B

def unembed_resid(model, l, h, c):
    rs = project_attn(model, l, h, c)
    logits = torch.stack([model.unembed(r) for r in rs])
    return torch.argmax(logits, dim=-1)

def calculate_attns(cache, l, h):
    prompts = cache.prompts
    model = cache.model
    input_tokens = model.to_tokens(prompts)
    batch, seq_len = input_tokens.shape[0], input_tokens.shape[1]
    
    q, k, v = decompose_head(cache, l, h)
    attn = cache['attn', l][:, h]
    qs = unembed_resid(cache.model, l, h, q.unsqueeze(2).expand(-1, -1, seq_len, -1))
    ks = unembed_resid(cache.model, l, h, k.unsqueeze(2).expand(-1, -1, seq_len, -1))
    vs = unembed_resid(cache.model, l, h, v.unsqueeze(2).expand(-1, -1, seq_len, -1))
    qk = unembed_resid(cache.model, l, h, q.unsqueeze(2) * k.unsqueeze(1))

    q_reshaped = q.unsqueeze(2).transpose(1, 2)
    k_reshaped = k.unsqueeze(1).transpose(1, 2)
    qkp = torch.einsum('bhse,bhte->bhst', q_reshaped, k_reshaped)
    qkp = qkp.transpose(1, 2).contiguous()
    qkp = unembed_resid(cache.model, l, h, qkp)

    layers = torch.full((batch, seq_len, seq_len), l).to(cache.device)
    heads = torch.full((batch, seq_len, seq_len), h).to(cache.device)

    data = torch.stack([
        layers, heads,
        input_tokens.unsqueeze(2).expand(-1, seq_len, seq_len),
        attn,
        qk, qkp, qs, ks, vs,
    ], dim=-1)

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(cache.device)
    mask = mask.unsqueeze(0).unsqueeze(-1)
    data = data.masked_fill(mask, -1)

    return data

def generate(cache):
    return torch.stack([
        calculate_attns(cache, *get_head_index(i))
        for i in range(cache.model.cfg.n_layers * cache.model.cfg.n_heads)
    ])

def to_df(data):
    num_layers, batch_size, seq_len, _, num_features = data.shape
    n = num_layers * batch_size * seq_len * seq_len
    
    df = pd.DataFrame(data.view(n, num_features).cpu())
    
    # Add layer, batch, and sequence position columns
    batch_indices = torch.arange(batch_size).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(num_layers, -1, seq_len, seq_len).flatten()
    seq_pos_indices_x = torch.arange(seq_len).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(num_layers, batch_size, seq_len, -1).flatten()
    seq_pos_indices_y = torch.arange(seq_len).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(num_layers, batch_size, -1, seq_len).flatten()
    
    df.insert(0, 'batch', batch_indices.cpu())
    df.insert(1, 'seq_pos_x', seq_pos_indices_x.cpu())
    df.insert(2, 'seq_pos_y', seq_pos_indices_y.cpu())
    df.columns = ['batch', 'pos_x', 'pos_y', 'layer', 'head', 'input', 'attn', 'qk', 'dp', 'q', 'k', 'v']
    return df

def df_to_tensor(df, layer, head):
    # Filter the DataFrame based on the given layer and head
    filtered_df = df[(df['layer'] == layer) & (df['head'] == head)]
    
    # Get the unique batch values
    batches = filtered_df['batch'].unique()
    
    # Get the maximum pos_x and pos_y values
    max_pos_x = filtered_df['pos_x'].max()
    max_pos_y = filtered_df['pos_y'].max()
    
    # Create an empty tensor with the desired shape
    tensor_shape = (len(batches), max_pos_x + 1, max_pos_y + 1, 9)
    tensor = torch.full(tensor_shape, -1)
    
    # Fill the tensor with the values from the DataFrame
    for _, row in filtered_df.iterrows():
        batch = int(row['batch'])
        pos_x = int(row['pos_x'])
        pos_y = int(row['pos_y'])
        features = torch.tensor(row[['batch', 'pos_x', 'input', 'attn', 'qk', 'dp', 'q', 'k', 'v']].values)
        tensor[batch, pos_y, pos_x] = features
    
    return tensor

def load(filepath):
    df = pd.read_csv(filepath)
    return df

def load_or_create_data(filename, model, *prompts):
    try:
        df = load(filename)
    except FileNotFoundError as e:
        cache = run_prompts(model, *prompts)
        print(f'Creating new file {filename}')
        data = generate(cache)
        df = to_df(data)
        df.to_csv(filename, index=False)
    return df

# PLOTTING FUNCTIONS

def plot_scatter(x, y, c, ax=None, cmap=cm.viridis, jitter_scale=0, s=5, alpha=0.7, third_dim=None):
    show_fig = False
    if ax is None:
        show_fig = True
        ax = plt
    
    # Add jitter to the x and y values if specified
    if jitter_scale > 0:
        jitter = np.random.normal(scale=jitter_scale, size=(len(x), 2))
        x += jitter[:, 0]
        y += jitter[:, 1]
    
    if third_dim is not None:
        ax.scatter(x, y, third_dim, c=c, cmap=cmap, s=3, alpha=alpha)
    else:
        ax.scatter(x, y, c=c, cmap=cmap, s=s, alpha=alpha)

    if show_fig:
        ax.tight_layout()
        ax.show()
    return ax

def plot_token_frequencies(model, token_counts, ax=None):
    plot_scatter(token_counts['Rank'], token_counts['Frequency'], token_counts['Head'], ax=ax)

def plot_unique_tokens_by_head(model, token_counts, ax=None):
    unique_token_counts = token_counts.groupby('head_index')['token'].nunique().reset_index(name='unique_tokens')
    plot_scatter(unique_token_counts['head'], unique_token_counts['unique_tokens'], unique_token_counts['layer'], ax=ax)

def plot_unique_tokens_by_layer_head(model, token_counts, ax=None):
    unique_token_counts = token_counts.groupby('Head')['Token'].nunique().reset_index(name='Unique Tokens')
    plot_scatter(unique_token_counts['head'], unique_token_counts['unique_tokens'], unique_token_counts['layer'], ax=ax)

def plot_token_embeddings(model, token_counts, embedding_method, colorbar_label='head_index', projection=2, ax=None, **kwargs):
    token_heads = token_counts[['token', colorbar_label]].drop_duplicates()
    
    token_embeddings = []
    for token in token_heads['token']:
        embedding = model.embed(token).detach().cpu().numpy()
        token_embeddings.append(embedding)
    
    if embedding_method == 'PCA':
        embedding_model = PCA(n_components=projection)
    elif embedding_method == 'TSNE':
        embedding_model = TSNE(n_components=projection, perplexity=kwargs.get('perplexity', 30),
                               learning_rate=kwargs.get('learning_rate', 200), random_state=42)
    elif embedding_method == 'UMAP':
        embedding_model = umap.UMAP(n_neighbors=kwargs.get('n_neighbors', 15), min_dist=kwargs.get('min_dist', 0.1),
                                    n_components=projection, random_state=42)
    else:
        raise ValueError(f"Unsupported embedding method: {embedding_method}")
    
    token_embeddings_transformed = embedding_model.fit_transform(np.stack(token_embeddings))
    df_embeddings = pd.DataFrame(token_embeddings_transformed, columns=[f'{embedding_method}{i+1}' for i in range(projection)])
    df_embeddings['token'] = token_heads['token']
    df_embeddings[colorbar_label] = token_heads[colorbar_label]
    
    third_dim = None if projection == 2 else df_embeddings[f'{embedding_method}3']
    plot_scatter(df_embeddings[f'{embedding_method}1'], df_embeddings[f'{embedding_method}2'], df_embeddings[colorbar_label], third_dim=third_dim, ax=ax)

def parse_tokens(tokens):
    for i in range(len(tokens)):
        t = tokens[i]
        if t == '\n':
            tokens[i] = '<nl>'
        if t == ' ':
            tokens[i] = '<sp>'
        if t == '\t':
            tokens[i] = '<tab>'
        if t == '<|endoftext|>':
            tokens[i] = 'EOS'
    return tokens

def add_attn_overlay(ax, pattern, threshold=0.3):
    seq_len = pattern.shape[1]
    for i in range(seq_len):
        for j in range(i + 1):
            if pattern[i, j] > threshold:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    linewidth=min(3, 2 + 5 * pattern[i, j].item()),
                    edgecolor='white',
                    facecolor='none',
                    alpha=min(1, pattern[i, j].item()),
                )
                ax.add_patch(rect)
    return ax

def add_axis_labels(model, ax, data, fontsize=12, feature_index=0):
    seq_len = data.shape[1]
    input_str_tokens = [
        model.to_single_str_token(int(t)) for t in torch.diag(data[feature_index, :, :, 2]).tolist()
    ]
    input_str_tokens = parse_tokens(input_str_tokens)

    q_str_tokens = [
        model.to_single_str_token(int(t)) for t in torch.diag(data[feature_index, :, :, 6]).tolist()
    ]
    q_str_tokens = parse_tokens(q_str_tokens)

    k_str_tokens = [
        model.to_single_str_token(int(t)) for t in torch.diag(data[feature_index, :, :, 7]).tolist()
    ]
    k_str_tokens = parse_tokens(k_str_tokens)

    v_str_tokens = [
        model.to_single_str_token(int(t)) for t in torch.diag(data[feature_index, :, :, 8]).tolist()
    ]
    v_str_tokens = parse_tokens(v_str_tokens)

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(k_str_tokens, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(input_str_tokens, fontsize=fontsize)
    ax.tick_params(axis='both', which='both', length=0)

    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(range(seq_len))
    ax2.set_yticklabels(v_str_tokens, fontsize=fontsize)
    ax2.tick_params(axis='y', which='both', length=0)

    ax3 = ax.secondary_xaxis('top')
    ax3.set_xticks(range(seq_len))
    ax3.set_xticklabels(q_str_tokens, fontsize=fontsize)
    ax3.tick_params(axis='x', which='both', length=0)

    return ax

def add_token_labels(model, ax, data, feature=4, feature_index=0, fontsize=10):
    seq_len = data.shape[1]
    tokens = data[feature_index, :, :, feature].tolist()
    str_tokens = [
        [model.to_single_str_token(int(t)) for t in row if t > -1] for row in tokens
    ]
    for i in range(seq_len):
        for j in range(i + 1):
            text = str_tokens[i][j]
            ax.text(j, i, text, ha='center', va='center', fontsize=fontsize, color='white', rotation=45)
    return ax

def unique_index_pattern(feature):
    unique_tokens = {}
    for x in feature:
        for y in x:
            value = y.item()
            if value == -1:
                unique_tokens[-1] = -len(unique_tokens)
            if value not in unique_tokens:
                unique_tokens[value] = len(unique_tokens)

    return [
        [unique_tokens[y.item()] for y in x] for x in feature
    ]

def plot_attn(model, attn_data, feature=4, feature_index=None, title=None, hide_labels=False, show_attn_overlay=True, show_axis=True, show_grid_labels=True, ax=None):
    attention = attn_data[:, :, :, 3]
    feature_data = attn_data[:, :, :, feature]
    if feature_index is None:
        feature_index = random.randint(0, len(feature_data) - 1)
    pattern = unique_index_pattern(feature_data[feature_index])

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.imshow(pattern)
    if not hide_labels and show_attn_overlay:
        ax = add_attn_overlay(ax, attention[feature_index], threshold=0)
    if not hide_labels and show_axis:
        ax = add_axis_labels(model, ax, attn_data, feature_index=feature_index)
    if not hide_labels and show_grid_labels:
        ax = add_token_labels(model, ax, attn_data, feature=feature, feature_index=feature_index)
    if title:
        ax.set_title(title)
    if hide_labels:
        ax.axis('off')
    
    return fig

def print_max_logits(cache, component='resid_post', layer=-1, k=5):
    resid_stream = cache[component, layer]
    resid_stream = cache.apply_ln_to_stack(resid_stream, layer)

    logits = cache.model.unembed(resid_stream)
    logits = cache.apply_ln_to_stack(logits, layer)

    top_pred_tokens = torch.topk(logits, k=k, dim=-1).indices.permute(2, 0, 1)
    tokens = cache.model.to_tokens(cache.prompts)
    pred_resid_directions = cache.model.tokens_to_residual_directions(top_pred_tokens[:, :, :-1])
    token_resid_directions = cache.model.tokens_to_residual_directions(tokens[:, 1:])
    data = torch.stack([
        (token_resid_directions - token_resid_directions).abs().mean((0, -1)),
        *(pred_resid_directions - token_resid_directions).abs().mean((1, -1)),
    ])
    fig = px.imshow(
        data.cpu(),
        color_continuous_midpoint=-0.5,
        color_continuous_scale="rdbu",
    )

    example_prompt = cache.model.to_str_tokens(cache.prompts[0])[1:] + ["..."]
    for x in range(len(example_prompt)):
        fig.add_annotation(x=x, y=0, text=example_prompt[x], showarrow=False, xshift=0, yshift=0, font=dict(size=16, color='white'))
        # use white if current cell is dark, black otherwise
        for i in range(k):
            color = 'black' if x == len(example_prompt) - 1 else 'white'
            fig.add_annotation(
                x=x, y=i + 1, 
                text=cache.model.tokenizer.decode(top_pred_tokens[i, 0, x]), 
                showarrow=False, xshift=0, yshift=0, font=dict(size=16, color=color),
            )

    fig.update_layout(
        width=1200, height=800,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    fig.update_coloraxes(showscale=False)
    return fig

def calculate_logit_diff(cache, completions):
    pred_tokens = torch.tensor([
        [cache.model.to_single_token(c) for c in completions] 
        for completions in completions 
    ]).to(cache.device)

    resid_directions = cache.model.tokens_to_residual_directions(pred_tokens)
    return resid_directions[:, 0] - resid_directions[:, 1]

def calculate_head_contribution(cache, towards, layer=-1, pos_slice=-1):
    per_head_residual = cache.stack_head_results(
        layer=layer, pos_slice=pos_slice,
    )

    per_head_logit_diffs = einsum(
        "... batch d_model, batch d_model -> ...",
        per_head_residual, towards,
    )

    return einops.rearrange(
        per_head_logit_diffs,
        "(layer head_index) -> layer head_index",
        layer=cache.model.cfg.n_layers,
        head_index=cache.model.cfg.n_heads,
    )

def plot_shared_token_heatmap(shared_token_df, token_type, ax=None, fig_size=(8, 6)):
    # Filter the DataFrame based on the specified token type
    token_type_col = f'{token_type}_count'
    filtered_df = shared_token_df[['head_index_2', 'head_index_1', token_type_col]]
    
    # Pivot the DataFrame to create a matrix of shared token counts
    pivot_df = filtered_df.pivot(index='head_index_2', columns='head_index_1', values=token_type_col)
    
    # Create a new figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    
    # Create a heatmap using seaborn
    sns.heatmap(pivot_df, annot=False, cmap='viridis', fmt='f', ax=ax)
    ax.set_title(token_type)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return ax

# SPECIFIC DATA ANALYSIS FUNCTIONS

def token_freq_data(model, df, feature):
    data = df.values
    feature_data = data[:, feature]
    tf = pd.DataFrame({'token': feature_data})
    tf = tf[tf['token'] != -1]
    tf['token'] = tf['token'].astype(int)
    tf = tf.groupby(['head', 'token']).size().reset_index(name='frequency')
    tf['token str'] = tf['token'].apply(model.to_single_str_token)
    tf['rank'] = tf.groupby('head')['frequency'].rank(method='dense', ascending=False)
    return tf

def calculate_token_frequencies(model, df, token_cols=['qk', 'dp', 'q', 'k', 'v']):
    # Melt the DataFrame to convert columns to rows
    melted_df = pd.melt(df, id_vars=['layer', 'head'], value_vars=token_cols, var_name='token_type', value_name='token')
    
    # Convert token column to integer and remove rows with -1 token values
    melted_df['token'] = melted_df['token'].astype(int)
    melted_df = melted_df[melted_df['token'] != -1]
    melted_df['head_index'] = ((melted_df['layer']) * 12) + (melted_df['head'])
    
    # Group by layer, head, token type, and token, and count the occurrences
    frequency_counts = melted_df.groupby(['head_index', 'token_type', 'token']).size().reset_index(name='count')
    
    # Pivot the DataFrame to have token types as columns and fill missing values with 0
    result = frequency_counts.pivot_table(index=['head_index', 'token'], columns='token_type', values='count', fill_value=0)
    result.columns = token_cols
    
    # Calculate the rank for each token type
    for col in token_cols:
        result[f'{col}_rank'] = result.groupby('head_index')[col].rank(method='dense', ascending=False)
    
    # Reset the index to convert layer, head, and token to regular columns
    result = result.reset_index()
    result['token str'] = result['token'].apply(model.to_single_str_token)
    
    return result

def calculate_shared_tokens(df, token_types=['qk', 'dp', 'q', 'k', 'v']):
    # Create an empty dictionary to store the shared tokens for each pair of heads and token type
    shared_tokens = {}

    # Get the unique head indices
    head_indices = df['head_index'].unique()

    # Iterate over each pair of head indices
    for i in range(len(head_indices)):
        for j in range(i + 1, len(head_indices)):
            head_index_1 = head_indices[i]
            head_index_2 = head_indices[j]

            # Initialize the shared token dictionary for the current pair of heads
            shared_tokens[(head_index_1, head_index_2)] = {}

            # Iterate over each token type
            for token_type in token_types:
                # Get the tokens for each head index and token type
                tokens_1 = df[(df['head_index'] == head_index_1) & (df[token_type] != 0)]['token'].values
                tokens_2 = df[(df['head_index'] == head_index_2) & (df[token_type] != 0)]['token'].values

                # Find the shared tokens between the two heads for the current token type
                shared = set(tokens_1) & set(tokens_2)

                # Store the shared tokens in the dictionary
                shared_tokens[(head_index_1, head_index_2)][token_type] = shared

    # Create a list to store the shared token counts and tokens
    shared_token_data = []

    # Iterate over the shared token dictionary and populate the list
    for (head_index_1, head_index_2), token_data in shared_tokens.items():
        row_data = {
            'head_index_1': head_index_1,
            'head_index_2': head_index_2
        }

        # Iterate over each token type and add the count and shared tokens to the row data
        for token_type in token_types:
            shared_tokens_type = token_data.get(token_type, set())
            row_data[f'{token_type}_count'] = len(shared_tokens_type)
            row_data[f'{token_type}_shared_tokens'] = ', '.join(map(str, shared_tokens_type))

        shared_token_data.append(row_data)

    # Create a DataFrame from the list of shared token data
    shared_token_df = pd.DataFrame(shared_token_data)

    return shared_token_df
