import random
import numpy as np
import torch
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from IPython.display import display, clear_output
pio.renderers.default = "png"


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

def project_attn(cache, l, h, c):
    OV = cache.model.OV[l, h]
    proj_A = torch.einsum('b n m h, d h -> b n m d', c, OV.A)
    proj_B = torch.einsum('b n m d, h d -> b n m d', proj_A, OV.B)
    return proj_B

def unembed_resid(cache, l, h, c):
    rs = project_attn(cache, l, h, c)
    logits = torch.stack([cache.model.unembed(r) for r in rs])
    return torch.argmax(logits, dim=-1)

def calculate_attns(cache, l, h):
    prompts = cache.prompts
    model = cache.model
    input_tokens = model.to_tokens(prompts)
    batch, seq_len = input_tokens.shape[0], input_tokens.shape[1]
    
    q, k, v = decompose_head(cache, l, h)
    attn = cache['attn', l][:, h]
    qs = unembed_resid(cache, l, h, q.unsqueeze(2).expand(-1, -1, seq_len, -1))
    ks = unembed_resid(cache, l, h, k.unsqueeze(2).expand(-1, -1, seq_len, -1))
    vs = unembed_resid(cache, l, h, v.unsqueeze(2).expand(-1, -1, seq_len, -1))
    qk = unembed_resid(cache, l, h, q.unsqueeze(2) * k.unsqueeze(1))
    q_reshaped = q.unsqueeze(2).transpose(1, 2)
    k_reshaped = k.unsqueeze(1).transpose(1, 2)
    qk = torch.einsum('bhse,bhte->bhst', q_reshaped, k_reshaped)
    qk = qk.transpose(1, 2).contiguous()
    qk = unembed_resid(cache, l, h, qk)
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
    
def head_index(i):
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

def plot_heads(cache, heads, **kwargs):
    plots = plot_attns(cache, heads, **kwargs)
    dropdown = Dropdown(
        options=[('Head {0}.{1}'.format(*head_index(i)), i) for i in range(len(plots))],
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
        calculate_attns(cache, *head_index(i))
        for i in range(cache.model.cfg.n_layers * cache.model.cfg.n_heads)
    ])

def to_df(data):
    n = torch.prod(torch.tensor([dim for dim in data.shape[:-1]]))
    return pd.DataFrame(data.view(n, -1).cpu())

def load(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['layer', 'head', 'Input token', 'attn', 'hp', 'q', 'k', 'v']
    return df

def token_freq_data(model, df, feature, shape):
    data = df.values
    feature_data = data[:, feature]
    tf = pd.DataFrame({'Token': feature_data})
    tf = tf[tf['Token'] != -1]
    tf['Head'] = tf.index // (shape[1] * shape[2] * shape[3])
    tf['Token'] = tf['Token'].astype(int)
    tf = tf.groupby(['Head', 'Token']).size().reset_index(name='Frequency')
    tf['Token str'] = tf['Token'].apply(model.to_single_str_token)
    tf['SortedIndex'] = tf.groupby('Head')['Frequency'].rank(method='dense', ascending=False)
    return tf

def plot_scatter(x, y, c, xlabel, ylabel, title=None, ax=None, cmap=cm.viridis, colorbar_label=None, jitter_scale=0, s=5, alpha=0.7):
    show_fig = False
    if ax is None:
        show_fig = True
        ax = plt
    
    # Add jitter to the x and y values if specified
    if jitter_scale > 0:
        jitter = np.random.normal(scale=jitter_scale, size=(len(x), 2))
        x += jitter[:, 0]
        y += jitter[:, 1]
    
    ax.scatter(x, y, c=c, cmap=cmap, s=s, alpha=alpha)
    if show_fig:
        ax.tight_layout()
        ax.show()
    return ax

def plot_token_frequencies(model, token_counts, **kwargs):
    plot_scatter(token_counts['SortedIndex'], token_counts['Frequency'], token_counts['Head'],
                 'Sorted Index', 'Frequency', 'Token Frequencies Grouped by Heads', **kwargs)

def plot_unique_tokens_by_head(model, token_counts, **kwargs):
    unique_token_counts = token_counts.groupby('Head')['Token'].nunique().reset_index(name='Unique Tokens')
    num_heads_per_layer = model.cfg.n_heads
    unique_token_counts['Layer'] = unique_token_counts['Head'] // num_heads_per_layer
    
    plot_scatter(unique_token_counts['Head'], unique_token_counts['Unique Tokens'], unique_token_counts['Layer'],
                 'Head Index', 'Count of Unique Tokens', 'Unique Token Counts by Head', **kwargs)

def plot_unique_tokens_by_layer_head(model, token_counts, **kwargs):
    unique_token_counts = token_counts.groupby('Head')['Token'].nunique().reset_index(name='Unique Tokens')
    num_heads_per_layer = model.cfg.n_heads
    unique_token_counts['Layer'] = unique_token_counts['Head'] // num_heads_per_layer
    unique_token_counts['Head Index within Layer'] = unique_token_counts['Head'] % num_heads_per_layer
    
    plot_scatter(unique_token_counts['Head Index within Layer'], unique_token_counts['Unique Tokens'], unique_token_counts['Layer'],
                 'Head Index within Layer', 'Count of Unique Tokens', 'Unique Token Counts by Layer and Head', **kwargs)

def plot_token_embeddings(model, token_counts, embedding_method, colorbar_label='Head', **kwargs):
    token_heads = token_counts[['Token', colorbar_label]].drop_duplicates()
    
    token_embeddings = []
    for token in token_heads['Token']:
        embedding = model.embed(token).detach().cpu().numpy()
        token_embeddings.append(embedding)
    
    if embedding_method == 'PCA':
        embedding_model = PCA(n_components=kwargs.get('num_components', 2))
    elif embedding_method == 'TSNE':
        embedding_model = TSNE(n_components=2, perplexity=kwargs.get('perplexity', 30),
                               learning_rate=kwargs.get('learning_rate', 200), random_state=42)
    elif embedding_method == 'UMAP':
        embedding_model = umap.UMAP(n_neighbors=kwargs.get('n_neighbors', 15), min_dist=kwargs.get('min_dist', 0.1),
                                    n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported embedding method: {embedding_method}")
    
    token_embeddings_transformed = embedding_model.fit_transform(np.stack(token_embeddings))
    
    df_embeddings = pd.DataFrame(token_embeddings_transformed, columns=[f'{embedding_method}{i+1}' for i in range(2)])
    df_embeddings['Token'] = token_heads['Token']
    df_embeddings[colorbar_label] = token_heads[colorbar_label]
    
    plot_scatter(df_embeddings[f'{embedding_method}1'], df_embeddings[f'{embedding_method}2'], df_embeddings[colorbar_label],
                 f'{embedding_method}1', f'{embedding_method}2', f'Token Embeddings {embedding_method}', **kwargs)

def add_max_labels(df, group_col, value_col, label_col, label_func):
    max_values = df.groupby(group_col)[value_col].max()
    for group, max_value in max_values.items():
        max_index = df[(df[group_col] == group) & (df[value_col] == max_value)].index[0]
        max_label = df.loc[max_index, label_col]
        label_text = f'({label_func(max_label)})'
        plt.text(max_label, max_value, label_text, fontsize=10, ha='left', va='bottom')

def figure(body, title=None, description=None, footer=None):
    content = []
    
    if title is not None:
        title_widget = HTML(f"<h2 style='font-size: {14}; text-align: center;'>{title}</h2>")
        content.append(title_widget)
    
    if description is not None:
        description_widget = HTML(f"<p style='font-size: {14}; text-align: center; width: 60%; margin: 0 auto;'>{description}</p>")
        content.append(description_widget)
        
    # Create an Output widget to capture the figure
    output = Output()

    # Display the figure in the Output widget
    with output:
        display(body)
    
    content.append(output)
    
    if footer is not None:
        footer_widget = HTML(f"<p style='font-size: {14}; text-align: center;'>{footer}</p>")
        content.append(footer_widget)
    
    return VBox(content)

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

def add_axis_labels(cache, ax, data, fontsize=12, feature_index=0):
    seq_len = data.shape[1]
    input_tokens = data[feature_index, -1, :, 0].int().tolist()
    input_str_tokens = [
        cache.model.to_single_str_token(t) for t in input_tokens
    ]
    input_str_tokens = parse_tokens(input_str_tokens)

    q_str_tokens = [
        cache.model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 5]).int().tolist()
    ]
    q_str_tokens = parse_tokens(q_str_tokens)

    k_str_tokens = [
        cache.model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 6]).int().tolist()
    ]
    k_str_tokens = parse_tokens(k_str_tokens)

    v_str_tokens = [
        cache.model.to_single_str_token(t) for t in torch.diag(data[feature_index, :, :, 7]).int().tolist()
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

def add_token_labels(cache, ax, data, feature=4, feature_index=0, fontsize=10):
    seq_len = data.shape[1]
    tokens = data[feature_index, :, :, feature].int().tolist()
    str_tokens = [
        [cache.model.to_single_str_token(t) for t in row if t > -1] for row in tokens
    ]
    for i in range(seq_len):
        for j in range(i + 1):
            text = str_tokens[i][j]
            ax.text(j, i, text, ha='center', va='center', fontsize=fontsize, color='white', rotation=45)
    return ax

def plot_attn(cache, attn_data, feature=4, feature_index=None, title=None, hide_labels=False, show_attn_overlay=True, show_axis=True, show_grid_labels=True, ax=None, **kwargs):
    attention = attn_data[:, :, :, 3]
    feature_data = attn_data[:, :, :, feature]
    if feature_index is None:
        feature_index = random.randint(0, len(feature_data) - 1)
    pattern = unique_index_pattern(feature_data[feature_index])

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.imshow(pattern)
    if not hide_labels and show_attn_overlay:
        ax = add_attn_overlay(ax, attention[feature_index], threshold=0)
    if not hide_labels and show_axis:
        ax = add_axis_labels(cache, ax, attn_data, feature_index=feature_index)
    if not hide_labels and show_grid_labels:
        ax = add_token_labels(cache, ax, attn_data, feature=feature, feature_index=feature_index)
    if title:
        ax.set_title(title)
    if hide_labels:
        ax.axis('off')
    
    return fig

def plot_attns(cache, heads, **kwargs):
    plots = [plot_attn(cache, *head_index(h), **kwargs) for h in heads]
    return plots