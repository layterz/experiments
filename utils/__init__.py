import random
import numpy as np
import torch
import math
from itertools import permutations, product
import einops
import pandas as pd
from fancy_einsum import einsum
from rich import print
from transformer_lens.utils import Slice
from transformer_lens import HookedTransformer
from plotly import express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from ipywidgets import Dropdown, Output, VBox, HBox, Layout, Label, HTML
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from IPython.display import display, clear_output

pio.renderers.default = "png"
plt.ioff()


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

def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def run_prompts(model: HookedTransformer, *prompts, **kwargs):
    device = get_device()
    _, cache = model.run_with_cache(list(prompts), **kwargs)
    cache.prompts = list(prompts)
    cache.device = device
    return cache

def hadamard_product_upto_position(q, k):
    # Create a mask to select elements up to and including the position
    mask = torch.tril(torch.ones(q.size(1), k.size(1))).to(q.device).unsqueeze(-1).unsqueeze(0)
    
    # Expand q and k to match the shape of the mask
    q_expanded = q.unsqueeze(2)
    k_expanded = k.unsqueeze(1)
    
    # Apply the mask to q_expanded and k_expanded
    q_masked = q_expanded * mask
    k_masked = k_expanded * mask
    
    # Take the Hadamard product of q_masked and k_masked along the last dimension
    result = q_masked * k_masked
    
    return result

def decompose_head(cache, l, h, pos=None):
    start, end = 0, len(cache['q', l][0])
    if pos is not None:
        start, end = pos
    q = cache['q', l][:, start:end, h, :]
    k = cache['k', l][:, start:end, h, :]
    v = cache['v', l][:, start:end, h, :]
    return torch.stack([q, k, v])

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
    qk = torch.einsum('bhse,bhte->bhst', q_reshaped, k_reshaped)
    qk = qk.transpose(1, 2).contiguous()
    qk = unembed_resid(cache.model, l, h, qk)
    layers = torch.full((batch, seq_len, seq_len), l).to(cache.device)
    heads = torch.full((batch, seq_len, seq_len), h).to(cache.device)

    data = torch.stack([
        layers, heads,
        input_tokens.unsqueeze(2).expand(-1, seq_len, seq_len),
        attn,
        qk, qs, ks, vs,
    ], dim=-1)

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(cache.device)
    mask = mask.unsqueeze(0).unsqueeze(-1)
    data = data.masked_fill(mask, -1)

    return data

def unique_index_pattern(feature):
    unique_tokens = {}
    for x in feature:
        for y in x:
            value = y.int().item()
            if value == -1:
                unique_tokens[-1] = -len(unique_tokens)
            if value not in unique_tokens:
                unique_tokens[value] = len(unique_tokens)

    return [
        [unique_tokens[y.int().item()] for y in x] for x in feature
    ]
    
def get_head_index(i):
    return (i // 12, i % 12)

def plot_layout(n):
    if n == 1:
        return {'width': 1000, 'height': 1000}
    if n == 2:
        return {'width': 600, 'height': 600}
    if n == 3:
        return {'width': 400, 'height': 400}
    if n == 4:
        return {'width': 350, 'height': 350}
    if n == 5:
        return {'width': 300, 'height': 300}
    
    raise Exception(f"Invalid grid shape: {n} plots.")

def plot_heads(model, heads, **kwargs):
    plots = plot_attns(model, heads, **kwargs)
    dropdown = Dropdown(
        options=[('Head {0}.{1}'.format(*get_head_index(i)), i) for i in range(len(plots))],
        value=0,
        description='Select Plot:',
    )
    output = Output()

    def update_plot(change):
        with output:
            clear_output(wait=True)  # Clear the previous plot
            # Directly display the selected plot
            plots[change['new']].show('png')

    dropdown.observe(update_plot, names='value')

    # Initialize with the first plot
    update_plot({'new': 0})

    return VBox([dropdown, output])

def generate(cache):
    return torch.stack([
        calculate_attns(cache, *get_head_index(i))
        for i in range(cache.model.cfg.n_layers * cache.model.cfg.n_heads)
    ])

def to_df(data):
    n = torch.prod(torch.tensor([dim for dim in data.shape[:-1]]))
    return pd.DataFrame(data.view(n, -1).cpu())

def load(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['layer', 'head', 'Input token', 'attn', 'hp', 'q', 'k', 'v']
    return df

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

def add_max_labels(df, group_col, value_col, label_col, label_func):
    max_values = df.groupby(group_col)[value_col].max()
    for group, max_value in max_values.items():
        max_index = df[(df[group_col] == group) & (df[value_col] == max_value)].index[0]
        max_label = df.loc[max_index, label_col]
        label_text = f'({label_func(max_label)})'
        plt.text(max_label, max_value, label_text, fontsize=10, ha='left', va='bottom')

def figure(*figs, rows=None, cols=None, figsize=None, title=None, footer=None):
    num_figs = len(figs)

    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(num_figs))
        rows = math.ceil(num_figs / cols)
    elif rows is None:
        rows = math.ceil(num_figs / cols)
    elif cols is None:
        cols = math.ceil(num_figs / rows)

    if figsize is None:
        figsize = (cols*6, rows*6)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, fig_obj in enumerate(figs):
        if i < num_figs:
            fig_obj.canvas.draw()
            plt.sca(axs[i])
            plt.imshow(fig_obj.canvas.renderer._renderer, cmap='viridis')
            plt.axis('off')
        else:
            axs[i].remove()

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    if footer is not None:
        fig.text(0.5, 0.05, footer, fontsize=12, ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

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
    input_tokens = data[feature_index, :, :, 2].int().tolist()
    input_str_tokens = [
        model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 2]).int().tolist()
    ]
    input_str_tokens = parse_tokens(input_str_tokens)

    q_str_tokens = [
        model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 5]).int().tolist()
    ]
    q_str_tokens = parse_tokens(q_str_tokens)

    k_str_tokens = [
        model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 6]).int().tolist()
    ]
    k_str_tokens = parse_tokens(k_str_tokens)

    v_str_tokens = [
        model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 7]).int().tolist()
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
    tokens = data[feature_index, :, :, feature].int().tolist()
    str_tokens = [
        [model.to_single_str_token(t) for t in row if t > -1] for row in tokens
    ]
    for i in range(seq_len):
        for j in range(i + 1):
            text = str_tokens[i][j]
            ax.text(j, i, text, ha='center', va='center', fontsize=fontsize, color='white', rotation=45)
    return ax

def plot_attn(model, attn_data, feature=4, feature_index=None, title=None, hide_labels=False, show_attn_overlay=True, show_axis=True, show_grid_labels=True, ax=None, **kwargs):
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

def plot_attns(model, heads, **kwargs):
    plots = [plot_attn(model, *get_head_index(h), **kwargs) for h in heads]
    return plots

def plot_grid(figs):
    display(figs)

def gallery(figs):
    def on_dropdown_change(change):
        layer, head = change['new'].split(',')
        layer = int(layer)
        head = int(head)
        fig = plot_attn(layer, head)
        output.clear_output(wait=True)
        with output:
            display(fig)

    # Create a dropdown widget
    options = [(f"Layer {layer}, Head {head}", f"{layer},{head}") for layer in range(12) for head in range(12)]
    dropdown = Dropdown(options=options, description="Select Layer and Head:")

    # Create an output widget to display the plot
    output = Output()

    # Observe changes in the dropdown value
    dropdown.observe(on_dropdown_change, names='value')

    # Display the dropdown and output widgets
    display(dropdown)
    display(output)

def shared_tokens(df, component='hp'):
    at = df[['layer', 'head', 'Input token', component, 'attn']]
    at.columns = ['layer', 'head', 'input_token', 'attention_token', 'attention_score']

    # Analyze subgroups for each attention head
    subgroup_data = []
    for layer, head in at[['layer', 'head']].drop_duplicates().itertuples(index=False):
        subgroup_mask = (at['layer'] == layer) & (at['head'] == head)
        unique_tokens = at.loc[subgroup_mask, 'attention_token'].unique()
        subgroup_data.append({
            'Layer': layer,
            'Head': head,
            'Unique Tokens': len(unique_tokens),
            'Tokens': ','.join(map(str, unique_tokens))
        })

    subgroups_df = pd.DataFrame(subgroup_data)

    # Analyze shared tokens between subgroups
    shared_token_data = []
    for i, (layer1, head1) in enumerate(subgroups_df[['Layer', 'Head']].itertuples(index=False)):
        for layer2, head2 in subgroups_df[['Layer', 'Head']].iloc[i+1:].itertuples(index=False):
            tokens1 = set(map(float, subgroups_df[(subgroups_df['Layer'] == layer1) & (subgroups_df['Head'] == head1)]['Tokens'].iloc[0].split(',')))
            tokens2 = set(map(float, subgroups_df[(subgroups_df['Layer'] == layer2) & (subgroups_df['Head'] == head2)]['Tokens'].iloc[0].split(',')))
            shared_tokens = tokens1.intersection(tokens2)
            if len(shared_tokens) > 0:
                shared_token_data.append({
                    'Subgroup 1': chr((int(layer1) * 12) + int(head1) + 65),
                    'Subgroup 2': chr((int(layer2) * 12) + int(head2) + 65),
                    'Shared Tokens': len(shared_tokens),
                    'Tokens': ','.join(map(str, shared_tokens))
                })

    return pd.DataFrame(shared_token_data)

def get_subgroup_label(layer_index, head_index):
    single_value = (layer_index * 12) + head_index
    return chr(single_value + ord('A'))

def visualize_shared_tokens(df, ax=None):
    # Assuming your data is stored in a DataFrame called 'df'
    # Extract the subgroup information and shared token counts from the DataFrame
    subgroup1 = df['Subgroup 1'].astype(str)
    subgroup2 = df['Subgroup 2'].sort_values(ascending=True).astype(str)
    shared_tokens = df['Shared Tokens'].astype(int)

    # Create a new DataFrame with subgroups as columns and shared token counts
    heatmap_data = pd.DataFrame({'Subgroup 1': subgroup1, 'Subgroup 2': subgroup2, 'Shared Tokens': shared_tokens})

    # Pivot the data to create a matrix
    shared_matrix = heatmap_data.pivot_table(index='Subgroup 2', columns='Subgroup 1', values='Shared Tokens', fill_value=0)

    # Create a figure and axes
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create the heatmap using Seaborn
    sns.heatmap(shared_matrix, annot=False, cmap='viridis', cbar_kws={'label': 'Shared Tokens'}, ax=ax)

    # Set the plot title and labels
    ax.set_title('Heatmap of Shared Tokens between Subgroups')
    ax.set_xlabel('Subgroup')
    ax.set_ylabel('Subgroup')
    ax.grid(which='major', linestyle='-', linewidth=0.5, color='white')

    num_ticks = 12  # Number of ticks to display
    step = len(shared_matrix) // num_ticks
    xtick_labels = [(i // 12, i % 12) for i in range(0, len(shared_matrix.columns), step)]
    ytick_labels = [(i // 12, i % 12) for i in range(0, len(shared_matrix.index), step)]

    ax.set_xticks(range(0, len(shared_matrix.columns), step))
    ax.set_xticklabels(xtick_labels)
    ax.set_yticks(range(0, len(shared_matrix.index), step))
    ax.set_yticklabels(ytick_labels)

    plt.xticks(rotation=-45, ha='left')
    plt.yticks(rotation=45, ha='right')

    # Display the plot
    plt.tight_layout()
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