import random
import numpy as np
import torch
from itertools import permutations, product
import einops
from fancy_einsum import einsum
from rich import print
from transformer_lens.utils import Slice
from plotly import express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from ipywidgets import Dropdown, Output, VBox, HBox, Layout, Label, HTML
from IPython.display import display


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
    logits = cache.model.unembed(rs[0])
    return torch.argmax(logits, dim=-1)

def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def run_prompts(model, *prompts):
    device = get_device()
    _, cache = model.run_with_cache(list(prompts))
    cache.prompts = list(prompts)
    cache.device = device
    return cache

def hadamard_product_upto_position(q, k):
    # Get the number of elements in the final dimension
    final_dim = q.size(-1)
    
    # Create a mask to select elements up to and including the position
    mask = torch.tril(torch.ones(final_dim, final_dim)).to(q.device).unsqueeze(0).unsqueeze(0)
    
    # Expand q and k to match the shape of the mask
    q_expanded = q.unsqueeze(2).expand(-1, -1, final_dim, -1)
    k_expanded = k.unsqueeze(1).expand(-1, final_dim, -1, -1)
    
    # Apply the mask to q_expanded and k_expanded
    q_masked = q_expanded * mask
    k_masked = k_expanded * mask
    
    # Take the Hadamard product of q_masked and k_masked along the last dimension
    result = q_masked * k_masked
    
    return result

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

def calculate_attns(cache, l, h):
    prompts = cache.prompts
    model = cache.model
    selected = random.randint(0, len(prompts) - 1)
    input_tokens = model.to_tokens(prompts[selected])
    seq_len = len(input_tokens[0])
    
    q, k, v = decompose_head(cache, l, h, pos=(0, seq_len))
    attn = cache['attn', l][selected, h]
    qs = unembed_resid(cache, l, h, q.unsqueeze(2).expand(-1, -1, seq_len, -1))
    ks = unembed_resid(cache, l, h, k.unsqueeze(2).expand(-1, -1, seq_len, -1))
    vs = unembed_resid(cache, l, h, v.unsqueeze(2).expand(-1, -1, seq_len, -1))
    hp = hadamard_product_upto_position(q, k)
    hp = unembed_resid(cache, l, h, hp)

    data = torch.stack([
        input_tokens.expand(seq_len, seq_len),
        attn,
        hp,
        qs,
        ks,
        vs,
    ], dim=-1)

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(cache.device)
    mask = mask.unsqueeze(0).unsqueeze(-1)
    data = data.masked_fill(mask, -1)

    return data

def outer_product(x, y):
    return q.unsqueeze(2) * k.unsqueeze(1)

def add_attn_overlay(cache, fig, data, min_attn=0.3):
    seq_len = data.shape[1]
    pattern = data[0, :, :, 1]
    for i in range(seq_len):
        for j in range(i + 1):
            if pattern[i, j] > min_attn:
                fig.add_shape(
                    type="rect",
                    x0=j-0.5,
                    y0=i-0.5,
                    x1=j+0.5,
                    y1=i+0.5,
                    line=dict(color="white", width=2),
                    fillcolor="rgba(0, 0, 0, 0)",
                )
    return fig

def add_axis_labels(cache, fig, data, fontsize=12, padding=20):
    seq_len = data.shape[1]
    input_tokens = data[0, -1, :, 0].int().tolist()
    input_str_tokens = [
        cache.model.to_single_str_token(t) for t in input_tokens
    ]
    input_str_tokens = parse_tokens(input_str_tokens)

    q_str_tokens = [
        cache.model.to_single_str_token(t) for t in torch.diag(data[0, :, :, 3]).int().tolist()
    ]
    q_str_tokens = parse_tokens(q_str_tokens)

    k_str_tokens = [
        cache.model.to_single_str_token(t) for t in torch.diag(data[0, :, :, 4]).int().tolist()
    ]
    k_str_tokens = parse_tokens(k_str_tokens)

    v_str_tokens = [
        cache.model.to_single_str_token(t) for t in torch.diag(data[0, :, :, 5]).int().tolist()
    ]
    v_str_tokens = parse_tokens(v_str_tokens)

    for i in range(seq_len):
        # top
        fig.add_annotation(x=i, y=-1, text=q_str_tokens[i], textangle=-90, showarrow=False, xshift=0, yshift=padding, font=dict(size=fontsize, color='black'))
        # right
        fig.add_annotation(x=seq_len, y=i, text=v_str_tokens[i], showarrow=False, xshift=padding, yshift=0, font=dict(size=fontsize, color='black'))
        # bottom
        fig.add_annotation(x=i, y=seq_len, text=k_str_tokens[i], textangle=-90, showarrow=False, xshift=0, yshift=-padding, font=dict(size=fontsize, color='black'))
        # left
        fig.add_annotation(x=-1, y=i, text=input_str_tokens[i], showarrow=False, xshift=-padding, yshift=0, font=dict(size=fontsize, color='black'))
    return fig

def add_token_labels(cache, fig, data, feature=2, fontsize=10):
    seq_len = data.shape[1]
    tokens = data[0, :, :, feature].int().tolist()
    str_tokens = [
        [cache.model.to_single_str_token(t) for t in row if t > -1] for row in tokens
    ]
    for i in range(seq_len):
        for j in range(i + 1):
            text = str_tokens[i][j]
            fig.add_annotation(
                x=j, y=i,
                text=text,
                textangle=-45,
                showarrow=False,
                font=dict(size=fontsize, color="white"),
            )
    return fig

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

def attn_grid(pattern, title=None, cmap='viridis'):
    fig = px.imshow(
        pattern,
        color_continuous_scale=cmap,
        color_continuous_midpoint=0.5,
        title=title,
    )

    fig.update_layout(
        width=1200, height=800,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='white',
    )
    
    fig.update_coloraxes(showscale=False)
    return fig

def plot_attn(cache, l, h, feature=2, hide_labels=False, show_attn_overlay=True, show_axis=True, show_grid_labels=True):
    data = calculate_attns(cache, l, h)
    feature = data[:, :, :, feature]
    unique_token_pattern = unique_index_pattern(feature[0])
    plot = attn_grid(unique_token_pattern)
    if not hide_labels and show_attn_overlay:
        plot = add_attn_overlay(cache, plot, data)
    if not hide_labels and show_axis:
        plot = add_axis_labels(cache, plot, data)
    if not hide_labels and show_grid_labels:
        plot = add_token_labels(cache, plot, data, 2)
    return plot

def head_index(i):
    return (i // 12, i % 12)

def plot_attns(cache, heads, **kwargs):
    plots = [plot_attn(cache, *head_index(h), **kwargs) for h in heads]
    return plots

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

def plot_grid(*plots, title=None, description=None, footer=None, rowsize=4):
    figure_layout = plot_layout(min(len(plots), rowsize))
    plot_widgets = [
        go.FigureWidget(
            p.update_layout(**figure_layout)
        ) for p in plots
    ]

    content = []
    if title is not None:
        title_widget = HTML(f"<h2 style='font-size: {14}; text-align: center;'>{title}</h2>")
        content.append(title_widget)
    
    if description is not None:
        description_widget = HTML(f"<p style='font-size: {14}; text-align: center;'>{description}</p>")
        content.append(description_widget)
    
    for i in range(0, len(plot_widgets), rowsize):
        content.append(HBox(plot_widgets[i:i+rowsize]))
    
    if footer is not None:
        footer_widget = HTML(f"<p style='font-size: {14}; text-align: center;'>{footer}</p>")
        content.append(footer_widget)
    display(VBox(content))