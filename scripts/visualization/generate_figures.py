#!/usr/bin/env python3
"""
=============================================================================
PUBLICATION-QUALITY FIGURES FOR SYLLOGISTIC REASONING BENCHMARK
=============================================================================
All 12 figures using Plotly with built-in templates and color sequences
Configuration: T=0.0, Few-Shot Prompting (unless comparing temperatures/strategies)
=============================================================================
"""

import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Using Plotly's Built-in Templates
# =============================================================================

# Get absolute path to project root
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_PATH = SCRIPT_DIR.parent
ANALYSIS_DIR = BASE_PATH / 'results' / 'analysis' / 'tables'
FIG_STATIC = BASE_PATH / 'results' / 'analysis' / 'figures' / 'static'
FIG_PLOTLY = BASE_PATH / 'results' / 'analysis' / 'figures' / 'plotly'

FIG_STATIC.mkdir(parents=True, exist_ok=True)
FIG_PLOTLY.mkdir(parents=True, exist_ok=True)

# Use Plotly's built-in template - 'plotly_white' is clean and modern
pio.templates.default = "plotly_white"

# Target configuration
TARGET_TEMP = 0.0
TARGET_STRATEGY = 'few_shot'

# 15 Complete Models
TARGET_MODELS = [
    'deepseek-v3.1', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro',
    'gemma-3-27b-it', 'glm-4.6', 'gpt-oss-20b', 'kimi-k2-instruct',
    'llama-3.1-8b-instruct', 'llama-3.2-1b-instruct', 'llama-3.2-3b-instruct',
    'llama-3.3-70b-instruct', 'mixtral-8x22b-instruct',
    'qwen3-next-80b-a3b-instruct', 'qwen3-next-80b-a3b-thinking'
]

def short_model_name(name):
    """Return full model name as-is for clarity in figures."""
    return name

def save_figure(fig, name, width=1200, height=700):
    """Save figure in HTML, PNG, and PDF formats."""
    # Save interactive HTML
    fig.write_html(FIG_PLOTLY / f'{name}.html', include_plotlyjs='cdn')
    
    # Save static images using kaleido
    try:
        fig.write_image(FIG_STATIC / f'{name}.png', width=width, height=height, scale=2, engine='kaleido')
        fig.write_image(FIG_STATIC / f'{name}.pdf', width=width, height=height, engine='kaleido')
        print(f"  ✅ Saved {name} (HTML + PNG + PDF)")
    except Exception as e:
        print(f"  ⚠️ Saved {name} (HTML only) - Static export error: {str(e)[:50]}")

# =============================================================================
# LOAD ALL DATA
# =============================================================================
print("\n" + "="*70)
print("LOADING DATA FILES")
print("="*70)

metrics_df = pd.read_csv(ANALYSIS_DIR / 'metrics_summary.csv')
variant_df = pd.read_csv(ANALYSIS_DIR / 'variant_accuracies.csv')
consistency_df = pd.read_csv(ANALYSIS_DIR / 'consistency_analysis.csv')
benchmark_df = pd.read_csv(ANALYSIS_DIR / 'benchmark_correlation_data.csv')
belief_bias_df = pd.read_csv(ANALYSIS_DIR / 'belief_bias_analysis.csv')
bootstrap_df = pd.read_csv(ANALYSIS_DIR / 'bootstrap_ci_accuracy.csv')
wilson_df = pd.read_csv(ANALYSIS_DIR / 'model_accuracy_wilson_CI.csv')
mcnemar_df = pd.read_csv(ANALYSIS_DIR / 'mcnemar_strategy_tests.csv')
wilcoxon_df = pd.read_csv(ANALYSIS_DIR / 'wilcoxon_strategy_pairs.csv')

with open(ANALYSIS_DIR / 'benchmark_correlations_corrected.json', 'r') as f:
    correlations = json.load(f)

print("✅ All data files loaded successfully")

# Filter for target configuration
config_df = metrics_df[
    (metrics_df['temperature'] == TARGET_TEMP) & 
    (metrics_df['strategy'] == TARGET_STRATEGY) &
    (metrics_df['model'].isin(TARGET_MODELS))
].copy()
config_df['syntax_pct'] = config_df['syntax_accuracy'] * 100
config_df['nlu_pct'] = config_df['nlu_accuracy'] * 100

print(f"✅ Filtered to {len(config_df)} models for T={TARGET_TEMP}, {TARGET_STRATEGY}")

# =============================================================================
# FIGURE 1: Model Performance Ranking (Horizontal Bar Chart)
# =============================================================================
print("\n" + "="*70)
print("FIGURE 1: Model Performance Ranking")
print("="*70)

df_sorted = config_df.sort_values('syntax_pct', ascending=True).reset_index(drop=True)
df_sorted['short_name'] = df_sorted['model'].apply(short_model_name)

fig1 = px.bar(
    df_sorted,
    y='short_name',
    x='syntax_pct',
    orientation='h',
    color='syntax_pct',
    color_continuous_scale='Viridis',
    text=[f"{v:.1f}%" for v in df_sorted['syntax_pct']],
    labels={'syntax_pct': 'Syntax Accuracy (%)', 'short_name': ''},
    title='<b>Model Performance on Syllogistic Reasoning</b><br><sup>Syntax Accuracy (%) | T=0.0, Few-Shot Prompting</sup>'
)

fig1.update_traces(textposition='outside', textfont_size=11)
fig1.update_layout(
    height=650, width=900,
    margin=dict(l=150, r=80, t=100, b=60),
    coloraxis_showscale=False,
    xaxis=dict(range=[0, 115], title='Syntax Accuracy (%)'),
    yaxis=dict(title=''),
    title_x=0.5
)

save_figure(fig1, 'fig01_model_ranking')

# =============================================================================
# FIGURE 2: Strategy Comparison Heatmap
# =============================================================================
print("\n" + "="*70)
print("FIGURE 2: Strategy Comparison Heatmap")
print("="*70)

strategy_df = metrics_df[
    (metrics_df['temperature'] == TARGET_TEMP) &
    (metrics_df['model'].isin(TARGET_MODELS))
].copy()

heatmap_data = strategy_df.pivot(index='model', columns='strategy', values='syntax_accuracy') * 100
heatmap_data['mean'] = heatmap_data.mean(axis=1)
heatmap_data = heatmap_data.sort_values('mean', ascending=False).drop('mean', axis=1)

col_order = ['zero_shot', 'one_shot', 'few_shot', 'zero_shot_cot']
col_labels = ['Zero-Shot', 'One-Shot', 'Few-Shot', 'Zero-Shot CoT']
heatmap_data = heatmap_data[col_order]
heatmap_data.columns = col_labels

fig2 = px.imshow(
    heatmap_data,
    color_continuous_scale='Viridis',
    aspect='auto',
    text_auto='.1f',
    labels=dict(x='Prompting Strategy', y='Model', color='Accuracy (%)'),
    title='<b>Strategy Performance Comparison</b><br><sup>Syntax Accuracy by Model and Prompting Strategy (T=0.0)</sup>'
)

fig2.update_layout(
    height=700, width=850,
    margin=dict(l=180, r=100, t=100, b=60),
    xaxis=dict(side='bottom'),
    yaxis=dict(ticktext=[short_model_name(m) for m in heatmap_data.index], tickvals=list(range(len(heatmap_data)))),
    title_x=0.5
)

save_figure(fig2, 'fig02_strategy_heatmap', width=850, height=700)

# =============================================================================
# FIGURE 3: Syntax vs NLU Butterfly Chart
# =============================================================================
print("\n" + "="*70)
print("FIGURE 3: Syntax vs NLU (Butterfly Chart)")
print("="*70)

var_config = variant_df[
    (variant_df['temperature'] == TARGET_TEMP) & 
    (variant_df['strategy'] == TARGET_STRATEGY) &
    (variant_df['model'].isin(TARGET_MODELS))
].copy()

var_config['syntax_mean'] = var_config[['N_acc', 'O_acc', 'X_acc', 'OX_acc']].mean(axis=1) * 100
nlu_data = config_df[['model', 'nlu_pct']].copy()
var_config = var_config.merge(nlu_data, on='model')
var_config = var_config.sort_values('syntax_mean', ascending=True)
var_config['short_name'] = var_config['model'].apply(short_model_name)

fig3 = go.Figure()

# Using Plotly's qualitative color sequence
colors = px.colors.qualitative.Plotly

fig3.add_trace(go.Bar(
    y=var_config['short_name'],
    x=var_config['syntax_mean'],
    orientation='h',
    name='Syntax Accuracy',
    marker_color=colors[0],
    text=[f"{v:.1f}%" for v in var_config['syntax_mean']],
    textposition='outside',
    textfont_size=9
))

fig3.add_trace(go.Bar(
    y=var_config['short_name'],
    x=-var_config['nlu_pct'],
    orientation='h',
    name='NLU Preference',
    marker_color=colors[1],
    text=[f"{v:.1f}%" for v in var_config['nlu_pct']],
    textposition='outside',
    textfont_size=9
))

fig3.update_layout(
    title='<b>Syntax vs NLU Preference Accuracy</b><br><sup>Diverging comparison across models (T=0.0, Few-Shot)</sup>',
    title_x=0.5,
    xaxis=dict(
        title='← NLU Preference (%)  |  Syntax Accuracy (%) →',
        tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
        ticktext=['100', '75', '50', '25', '0', '25', '50', '75', '100'],
        range=[-110, 110],
        zeroline=True, zerolinewidth=2, zerolinecolor='gray'
    ),
    yaxis=dict(title=''),
    barmode='overlay',
    height=650, width=1000,
    margin=dict(l=150, r=80, t=100, b=80),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)

save_figure(fig3, 'fig03_syntax_vs_nlu_butterfly')

# =============================================================================
# FIGURE 4: Variant-wise Accuracy (Scatter with Model Identification)
# =============================================================================
print("\n" + "="*70)
print("FIGURE 4: Accuracy by Variant Type")
print("="*70)

# Correct labels based on dataset documentation
variant_cols = ['N_acc', 'O_acc', 'X_acc', 'OX_acc']
variant_labels_map = {
    'N_acc': 'Normal (N)',
    'O_acc': 'Order-Switched (O)', 
    'X_acc': 'Nonsense (X)',
    'OX_acc': 'Nonsense+Order (OX)'
}
variant_order = ['Normal (N)', 'Order-Switched (O)', 'Nonsense (X)', 'Nonsense+Order (OX)']

# Reshape data - each model becomes a data point per variant
var_long = pd.melt(
    var_config[['model'] + variant_cols],
    id_vars=['model'],
    var_name='Variant',
    value_name='Accuracy'
)
var_long['Accuracy'] = var_long['Accuracy'] * 100
var_long['Variant'] = var_long['Variant'].map(variant_labels_map)
var_long['short_name'] = var_long['model'].apply(short_model_name)

# Calculate means to determine the "winner"
variant_means = var_long.groupby('Variant')['Accuracy'].mean()
best_variant = variant_means.idxmax()

# Use Plotly's built-in color sequence and marker symbols
model_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
model_markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
                 'pentagon', 'hexagon', 'star', 'hourglass', 'bowtie', 'circle-open', 'square-open', 'diamond-open']

# Get unique models sorted
unique_models = sorted(var_long['short_name'].unique())
model_to_color = {m: model_colors[i % len(model_colors)] for i, m in enumerate(unique_models)}
model_to_marker = {m: model_markers[i % len(model_markers)] for i, m in enumerate(unique_models)}

fig4 = go.Figure()

# First add box plots (no points) with light grey for context
for variant in variant_order:
    variant_data = var_long[var_long['Variant'] == variant]
    is_best = (variant == best_variant)
    
    fig4.add_trace(go.Box(
        y=variant_data['Accuracy'],
        x=[variant] * len(variant_data),
        name=variant,
        marker_color='rgba(99, 110, 250, 0.3)' if is_best else 'rgba(150, 150, 150, 0.3)',
        line=dict(color='rgba(99, 110, 250, 0.6)' if is_best else 'rgba(150, 150, 150, 0.6)', width=1.5),
        fillcolor='rgba(99, 110, 250, 0.1)' if is_best else 'rgba(200, 200, 200, 0.1)',
        boxpoints=False,  # We'll add our own scatter points
        showlegend=False
    ))

# Add scatter points for each model with distinct colors/markers
for model_name in unique_models:
    model_data = var_long[var_long['short_name'] == model_name]
    
    # Add jitter to x positions
    x_positions = []
    for v in model_data['Variant']:
        base_pos = variant_order.index(v)
        jitter = np.random.uniform(-0.25, 0.25)
        x_positions.append(base_pos + jitter)
    
    fig4.add_trace(go.Scatter(
        x=[variant_order[int(round(x))] for x in x_positions],  # Use category names
        y=model_data['Accuracy'],
        mode='markers',
        name=model_name,
        marker=dict(
            symbol=model_to_marker[model_name],
            size=10,
            color=model_to_color[model_name],
            line=dict(width=1, color='white'),
            opacity=0.85
        ),
        legendgroup=model_name,
        hovertemplate=f'<b>{model_name}</b><br>Variant: %{{x}}<br>Syntax Accuracy: %{{y:.1f}}%<extra></extra>'
    ))

# Add mean annotations ABOVE the boxes (at the top of the plot)
max_acc = var_long['Accuracy'].max()
for variant in variant_order:
    mean_val = variant_means[variant]
    fig4.add_annotation(
        x=variant,
        y=max_acc + 8,  # Position above the highest data point
        text=f"μ={mean_val:.1f}%",
        showarrow=False,
        font=dict(size=11, color='#333', weight='bold')
    )

fig4.update_layout(
    title=f"<b>Syntax Accuracy Distribution by Syllogism Variant</b><br><sup>{best_variant} shows highest mean | T=0.0, Few-Shot Prompting | N=15 models</sup>",
    title_x=0.5,
    template='simple_white',
    height=600, width=1100,
    margin=dict(l=80, r=220, t=100, b=80),
    yaxis=dict(title='Syntax Accuracy (%)', range=[40, 115]),  # Extended range for μ labels
    xaxis=dict(title='Syllogism Variant', categoryorder='array', categoryarray=variant_order),
    legend=dict(
        orientation='v',
        yanchor='middle',
        y=0.5,
        xanchor='left',
        x=1.02,
        font=dict(size=9),
        title=dict(text='<b>Models</b>', font=dict(size=10), side='top'),
        itemsizing='constant',
        tracegroupgap=0
    ),
    hovermode='closest'
)

save_figure(fig4, 'fig04_variant_accuracy', width=1100, height=600)

# =============================================================================
# FIGURE 5: LMArena Correlation (was Figure 6)
# =============================================================================
print("\n" + "="*70)
print("FIGURE 5: Correlation with LMArena Rank")
print("="*70)

bench_merged = benchmark_df.dropna(subset=['lmarena_rank']).copy()
bench_merged['syllogism_pct'] = bench_merged['syllogism_accuracy'] * 100
bench_merged['short_name'] = bench_merged['model'].apply(short_model_name)

lmar_corr = correlations['lmarena']
r_val, p_val = lmar_corr['spearman_r'], lmar_corr['spearman_p']

# Define distinct markers for each model
markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
           'pentagon', 'hexagon', 'star', 'hourglass', 'bowtie', 'circle-open', 'square-open', 'diamond-open']
colors = px.colors.qualitative.Dark24

fig6 = go.Figure()

# Add individual scatter points with unique marker and color per model
for i, (_, row) in enumerate(bench_merged.iterrows()):
    fig6.add_trace(go.Scatter(
        x=[row['lmarena_rank']],
        y=[row['syllogism_pct']],
        mode='markers',
        name=row['short_name'],
        marker=dict(
            symbol=markers[i % len(markers)],
            size=14,
            color=colors[i % len(colors)],
            line=dict(width=1, color='white')
        ),
        showlegend=True,
        hovertemplate=f"<b>{row['short_name']}</b><br>LMArena Rank: {row['lmarena_rank']:.0f}<br>Syntax Acc: {row['syllogism_pct']:.1f}%<extra></extra>"
    ))

# Add OLS trendline with confidence interval
from scipy import stats
import statsmodels.api as sm

X = bench_merged['lmarena_rank'].values
Y = bench_merged['syllogism_pct'].values
X_with_const = sm.add_constant(X)
model = sm.OLS(Y, X_with_const).fit()
predictions = model.get_prediction(X_with_const)
pred_summary = predictions.summary_frame(alpha=0.05)

# Sort for plotting
sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
Y_pred = pred_summary['mean'].values[sort_idx]
Y_ci_low = pred_summary['obs_ci_lower'].values[sort_idx]
Y_ci_high = pred_summary['obs_ci_upper'].values[sort_idx]

# Add confidence interval band
fig6.add_trace(go.Scatter(
    x=np.concatenate([X_sorted, X_sorted[::-1]]),
    y=np.concatenate([Y_ci_high, Y_ci_low[::-1]]),
    fill='toself',
    fillcolor='rgba(99, 102, 241, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False,
    name='95% CI',
    hoverinfo='skip'
))

# Add regression line
fig6.add_trace(go.Scatter(
    x=X_sorted,
    y=Y_pred,
    mode='lines',
    line=dict(color='#ef4444', width=2, dash='solid'),
    name=f'OLS (ρ={r_val:.2f})',
    showlegend=True
))

fig6.update_layout(
    title=f'<b>Syntax Accuracy vs LMArena Rank</b><br><sup>Spearman ρ = {r_val:.3f}, p = {p_val:.4f} | Lower rank = better</sup>',
    title_x=0.5,
    xaxis=dict(title='LMArena Rank (lower is better)', showgrid=True, gridcolor='lightgray', gridwidth=1),
    yaxis=dict(title='Syntax Accuracy (%) [T=0.0, Avg. all strategies]', range=[40, 110], showgrid=True, gridcolor='lightgray', gridwidth=1),
    height=600, width=950,
    margin=dict(l=100, r=200, t=100, b=80),
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=10)
    ),
    plot_bgcolor='white'
)

save_figure(fig6, 'fig05_lmarena_correlation', width=950, height=600)

# =============================================================================
# FIGURE 6: MMLU Correlation (was Figure 7)
# =============================================================================
print("\n" + "="*70)
print("FIGURE 6: Correlation with MMLU")
print("="*70)

mmlu_merged = benchmark_df.dropna(subset=['mmlu_score']).copy()
mmlu_merged['syllogism_pct'] = mmlu_merged['syllogism_accuracy'] * 100
mmlu_merged['mmlu_pct'] = mmlu_merged['mmlu_score'] * 100
mmlu_merged['short_name'] = mmlu_merged['model'].apply(short_model_name)

mmlu_corr = correlations['mmlu']
r_mmlu, p_mmlu = mmlu_corr['spearman_r'], mmlu_corr['spearman_p']

fig7 = go.Figure()

# Add individual scatter points with unique marker and color per model
for i, (_, row) in enumerate(mmlu_merged.iterrows()):
    fig7.add_trace(go.Scatter(
        x=[row['mmlu_pct']],
        y=[row['syllogism_pct']],
        mode='markers',
        name=row['short_name'],
        marker=dict(
            symbol=markers[i % len(markers)],
            size=14,
            color=colors[i % len(colors)],
            line=dict(width=1, color='white')
        ),
        showlegend=True,
        hovertemplate=f"<b>{row['short_name']}</b><br>MMLU: {row['mmlu_pct']:.1f}%<br>Syntax Acc: {row['syllogism_pct']:.1f}%<extra></extra>"
    ))

# Add OLS trendline with confidence interval
X_mmlu = mmlu_merged['mmlu_pct'].values
Y_mmlu = mmlu_merged['syllogism_pct'].values
X_mmlu_const = sm.add_constant(X_mmlu)
model_mmlu = sm.OLS(Y_mmlu, X_mmlu_const).fit()
pred_mmlu = model_mmlu.get_prediction(X_mmlu_const)
pred_summary_mmlu = pred_mmlu.summary_frame(alpha=0.05)

sort_idx_mmlu = np.argsort(X_mmlu)
X_mmlu_sorted = X_mmlu[sort_idx_mmlu]
Y_pred_mmlu = pred_summary_mmlu['mean'].values[sort_idx_mmlu]
Y_ci_low_mmlu = pred_summary_mmlu['obs_ci_lower'].values[sort_idx_mmlu]
Y_ci_high_mmlu = pred_summary_mmlu['obs_ci_upper'].values[sort_idx_mmlu]

# Add confidence interval band
fig7.add_trace(go.Scatter(
    x=np.concatenate([X_mmlu_sorted, X_mmlu_sorted[::-1]]),
    y=np.concatenate([Y_ci_high_mmlu, Y_ci_low_mmlu[::-1]]),
    fill='toself',
    fillcolor='rgba(16, 185, 129, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False,
    name='95% CI',
    hoverinfo='skip'
))

# Add regression line
fig7.add_trace(go.Scatter(
    x=X_mmlu_sorted,
    y=Y_pred_mmlu,
    mode='lines',
    line=dict(color='#10b981', width=2, dash='solid'),
    name=f'OLS (ρ={r_mmlu:.2f})',
    showlegend=True
))

fig7.update_layout(
    title=f'<b>Syntax Accuracy vs MMLU Score</b><br><sup>Spearman ρ = {r_mmlu:.3f}, p = {p_mmlu:.3f} (NOT significant)</sup>',
    title_x=0.5,
    xaxis=dict(title='MMLU Score (%)', showgrid=True, gridcolor='lightgray', gridwidth=1),
    yaxis=dict(title='Syntax Accuracy (%) [T=0.0, Avg. all strategies]', range=[40, 110], showgrid=True, gridcolor='lightgray', gridwidth=1),
    height=600, width=950,
    margin=dict(l=100, r=200, t=100, b=80),
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=10)
    ),
    plot_bgcolor='white'
)

save_figure(fig7, 'fig06_mmlu_correlation', width=950, height=600)

# =============================================================================
# FIGURE 5-6 COMBINED: Stitch fig6 and fig7 side by side
# =============================================================================
print("\n" + "="*70)
print("FIGURE 5-6 COMBINED: Benchmark Correlations (Side by Side)")
print("="*70)

# Create combined figure by stitching fig6 (LMArena) and fig7 (MMLU)
fig_combined = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f'<b>vs LMArena Rank</b><br><sup>ρ = {r_val:.3f}, p = {p_val:.4f}</sup>',
        f'<b>vs MMLU Score</b><br><sup>ρ = {r_mmlu:.3f}, p = {p_mmlu:.3f} (n.s.)</sup>'
    ],
    horizontal_spacing=0.12
)

# Copy all traces from fig6 to left subplot (col=1)
for trace in fig6.data:
    new_trace = trace
    # Only show legend for scatter points (models), not CI band or regression line
    if hasattr(trace, 'name') and trace.name and trace.name not in ['95% CI', ''] and 'OLS' not in str(trace.name):
        new_trace.showlegend = True
    else:
        new_trace.showlegend = False
    fig_combined.add_trace(new_trace, row=1, col=1)

# Copy all traces from fig7 to right subplot (col=2)
# Track which models already have legend entries from fig6
fig6_models = set()
for trace in fig6.data:
    if hasattr(trace, 'name') and trace.name:
        fig6_models.add(trace.name)

for trace in fig7.data:
    new_trace = trace
    # Only show in legend if not already shown from fig6
    if hasattr(trace, 'name') and trace.name and trace.name not in ['95% CI', ''] and 'OLS' not in str(trace.name):
        if trace.name in fig6_models:
            new_trace.showlegend = False
        else:
            new_trace.showlegend = True
    else:
        new_trace.showlegend = False
    fig_combined.add_trace(new_trace, row=1, col=2)

fig_combined.update_layout(
    title=dict(
        text='<b>Benchmark Correlations: Syllogistic Reasoning vs External Benchmarks</b>',
        x=0.5,
        font=dict(size=16)
    ),
    height=550,
    width=1400,
    margin=dict(l=80, r=250, t=100, b=80),
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=10),
        title=dict(text='<b>Models</b>', font=dict(size=11)),
        itemsizing='constant'
    ),
    plot_bgcolor='white'
)

# Update axes for both subplots
fig_combined.update_xaxes(title_text='LMArena Rank (lower = better)', showgrid=True, gridcolor='lightgray', row=1, col=1)
fig_combined.update_xaxes(title_text='MMLU Score (%)', showgrid=True, gridcolor='lightgray', row=1, col=2)
fig_combined.update_yaxes(title_text='Syntax Accuracy (%) [T=0.0, Avg.]', range=[40, 110], showgrid=True, gridcolor='lightgray', row=1, col=1)
fig_combined.update_yaxes(title_text='Syntax Accuracy (%) [T=0.0, Avg.]', range=[40, 110], showgrid=True, gridcolor='lightgray', row=1, col=2)

save_figure(fig_combined, 'fig05_06_benchmark_correlations_combined', width=1400, height=550)

# =============================================================================
# FIGURE 7: Belief Bias Analysis - Extended Dumbbell Plot
# =============================================================================
print("\n" + "="*70)
print("FIGURE 7: Belief Bias Analysis (Dumbbell Plot)")
print("="*70)

bias_df = belief_bias_df.copy()
bias_df['congruent_pct'] = bias_df['avg_congruent_acc'] * 100
bias_df['incongruent_pct'] = bias_df['avg_incongruent_acc'] * 100
bias_df['bias_gap'] = bias_df['congruent_pct'] - bias_df['incongruent_pct']  # The "Belief Bias"
bias_df['short_name'] = bias_df['model'].apply(short_model_name)

# Sort by bias gap (largest bias at top for visual impact)
bias_df = bias_df.sort_values('bias_gap', ascending=True).reset_index(drop=True)

fig8 = go.Figure()

# Add alternating background shading for readability
for i in range(len(bias_df)):
    if i % 2 == 0:
        fig8.add_shape(
            type="rect",
            x0=0, x1=105,
            y0=i - 0.4, y1=i + 0.4,
            fillcolor="rgba(240, 240, 240, 0.5)",
            line=dict(width=0),
            layer="below"
        )

# Add connecting lines (the "gap" visualization) - drawn first so dots appear on top
for i, row in bias_df.iterrows():
    idx = bias_df.index.get_loc(i) if isinstance(bias_df.index, pd.RangeIndex) else list(bias_df.index).index(i)
    # Determine line color based on gap size
    gap = row['bias_gap']
    if gap > 15:
        line_color = 'rgba(239, 68, 68, 0.6)'  # Red for high bias
    elif gap > 5:
        line_color = 'rgba(251, 191, 36, 0.6)'  # Yellow/amber for medium bias
    else:
        line_color = 'rgba(34, 197, 94, 0.6)'  # Green for low bias
    
    fig8.add_trace(go.Scatter(
        x=[row['incongruent_pct'], row['congruent_pct']],
        y=[row['short_name'], row['short_name']],
        mode='lines',
        line=dict(color=line_color, width=8),
        hoverinfo='skip',
        showlegend=False
    ))

# Add "Incongruent" points (harder task - typically lower)
fig8.add_trace(go.Scatter(
    x=bias_df['incongruent_pct'],
    y=bias_df['short_name'],
    mode='markers',
    name='Incongruent (Harder)',
    marker=dict(
        color='#dc2626',  # Red
        size=14,
        line=dict(width=2, color='white'),
        symbol='circle'
    ),
    hovertemplate='<b>%{y}</b><br>Incongruent: %{x:.1f}%<extra></extra>'
))

# Add "Congruent" points (easier task - typically higher)
fig8.add_trace(go.Scatter(
    x=bias_df['congruent_pct'],
    y=bias_df['short_name'],
    mode='markers',
    name='Congruent (Easier)',
    marker=dict(
        color='#2563eb',  # Blue
        size=14,
        line=dict(width=2, color='white'),
        symbol='circle'
    ),
    hovertemplate='<b>%{y}</b><br>Congruent: %{x:.1f}%<extra></extra>'
))

# Add delta (bias gap) annotations floating on the connecting lines
for i, row in bias_df.iterrows():
    gap = row['bias_gap']
    midpoint_x = (row['congruent_pct'] + row['incongruent_pct']) / 2
    
    # Color the annotation based on gap severity
    if gap > 15:
        text_color = '#dc2626'  # Red
        gap_text = f"Δ{gap:.1f}%"
    elif gap > 5:
        text_color = '#d97706'  # Amber
        gap_text = f"Δ{gap:.1f}%"
    elif gap < -5:
        text_color = '#059669'  # Green (negative = incongruent better, rare)
        gap_text = f"Δ{gap:.1f}%"
    else:
        text_color = '#059669'  # Green for small gap
        gap_text = f"Δ{gap:.1f}%"
    
    fig8.add_annotation(
        x=midpoint_x,
        y=row['short_name'],
        text=f"<b>{gap_text}</b>",
        showarrow=False,
        font=dict(size=10, color=text_color),
        bgcolor='white',
        borderpad=2,
        yshift=0
    )

# Add interpretation zones as vertical bands
fig8.add_shape(
    type="rect",
    x0=0, x1=50,
    y0=-0.5, y1=len(bias_df) - 0.5,
    fillcolor="rgba(254, 202, 202, 0.15)",  # Light red
    line=dict(width=0),
    layer="below"
)
fig8.add_shape(
    type="rect",
    x0=50, x1=80,
    y0=-0.5, y1=len(bias_df) - 0.5,
    fillcolor="rgba(254, 249, 195, 0.15)",  # Light yellow
    line=dict(width=0),
    layer="below"
)
fig8.add_shape(
    type="rect",
    x0=80, x1=105,
    y0=-0.5, y1=len(bias_df) - 0.5,
    fillcolor="rgba(187, 247, 208, 0.15)",  # Light green
    line=dict(width=0),
    layer="below"
)

# Add zone labels at bottom
fig8.add_annotation(x=25, y=-0.8, text="<b>Low Accuracy</b>", showarrow=False, 
                    font=dict(size=9, color='#991b1b'), yref='y')
fig8.add_annotation(x=65, y=-0.8, text="<b>Medium</b>", showarrow=False,
                    font=dict(size=9, color='#92400e'), yref='y')
fig8.add_annotation(x=92, y=-0.8, text="<b>High Accuracy</b>", showarrow=False,
                    font=dict(size=9, color='#166534'), yref='y')

fig8.update_layout(
    title='<b>Belief Bias Analysis: The Gap in Logical Reasoning</b><br><sup>T=0.0 | Avg. across all strategies | Syntax Accuracy | Sorted by bias magnitude</sup>',
    title_x=0.5,
    template='simple_white',
    height=650, width=1000,
    margin=dict(l=220, r=80, t=120, b=100),
    xaxis=dict(
        title='Accuracy (%)',
        range=[0, 105],
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        zeroline=False
    ),
    yaxis=dict(
        title='',
        showgrid=False,
        categoryorder='array',
        categoryarray=bias_df['short_name'].tolist()
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=11)
    ),
    hovermode='closest'
)

save_figure(fig8, 'fig07_belief_bias', width=1000, height=650)

# =============================================================================
# FIGURE 8: Confidence Intervals (was Figure 9)
# =============================================================================
print("\n" + "="*70)
print("FIGURE 8: Model Accuracy with Confidence Intervals")
print("="*70)

ci_df = bootstrap_df.copy()
ci_df = ci_df.sort_values('accuracy', ascending=True)
ci_df['short_name'] = ci_df['model'].apply(short_model_name)
ci_df['error_low'] = ci_df['accuracy'] - ci_df['ci_lower']
ci_df['error_high'] = ci_df['ci_upper'] - ci_df['accuracy']

fig9 = go.Figure()

fig9.add_trace(go.Scatter(
    y=ci_df['short_name'],
    x=ci_df['accuracy'],
    mode='markers',
    marker=dict(
        size=12,
        color=ci_df['accuracy'],
        colorscale='Viridis',
        line=dict(width=1, color='white')
    ),
    error_x=dict(
        type='data',
        symmetric=False,
        array=ci_df['error_high'],
        arrayminus=ci_df['error_low'],
        thickness=2,
        width=6,
        color='gray'
    ),
    hovertemplate='<b>%{y}</b><br>Accuracy: %{x:.1f}%<br>CI: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<extra></extra>',
    customdata=np.stack([ci_df['ci_lower'], ci_df['ci_upper']], axis=1)
))

fig9.update_layout(
    title='<b>Model Accuracy with 95% Bootstrap CIs</b><br><sup>Error bars show confidence interval bounds</sup>',
    title_x=0.5,
    xaxis=dict(title='Accuracy (%)', range=[30, 110]),
    yaxis=dict(title=''),
    height=650, width=900,
    margin=dict(l=150, r=80, t=100, b=60)
)

save_figure(fig9, 'fig08_confidence_intervals')

# =============================================================================
# FIGURE 9: McNemar Test Results (Improved Visualization, was Figure 11)
# =============================================================================
print("\n" + "="*70)
print("FIGURE 9: Statistical Significance (McNemar Tests)")
print("="*70)

mcnemar_plot = mcnemar_df.copy()
mcnemar_plot['neg_log_p'] = -np.log10(mcnemar_plot['p_value'].clip(lower=1e-10))

# Clean up comparison labels for readability
mcnemar_plot['comparison_clean'] = (mcnemar_plot['comparison']
    .str.replace('_vs_', ' vs ')
    .str.replace('_', '-')
    .str.title()
    .str.replace('Cot', 'CoT'))

# Sort by significance (most significant at top for visual impact)
mcnemar_plot = mcnemar_plot.sort_values('neg_log_p', ascending=True)

# Format p-values for annotations (show actual p-value, not log)
def format_pvalue(p):
    if p < 0.0001:
        return f"p < 0.0001"
    elif p < 0.001:
        return f"p = {p:.5f}"
    else:
        return f"p = {p:.4f}"

mcnemar_plot['p_label'] = mcnemar_plot['p_value'].apply(format_pvalue)

# Color scheme: Green for significant (stands out), Gray for not significant (fades)
# Using softer, more modern colors
sig_color = '#10b981'  # Emerald green - significant
nonsig_color = '#9ca3af'  # Gray - not significant

fig11 = go.Figure()

# Add bars manually for better control
for idx, row in mcnemar_plot.iterrows():
    bar_color = sig_color if row['significant'] else nonsig_color
    
    fig11.add_trace(go.Bar(
        y=[row['comparison_clean']],
        x=[row['neg_log_p']],
        orientation='h',
        marker=dict(
            color=bar_color,
            line=dict(width=0)
        ),
        name='Significant' if row['significant'] else 'Not Significant',
        showlegend=False,
        hovertemplate=(
            f"<b>{row['comparison_clean']}</b><br>"
            f"p-value: {row['p_value']:.6f}<br>"
            f"-log₁₀(p): {row['neg_log_p']:.2f}<br>"
            f"χ²: {row['chi2']:.2f}<extra></extra>"
        )
    ))

# Add p-value annotations on the bars (the "intuition fix")
for idx, row in mcnemar_plot.iterrows():
    fig11.add_annotation(
        x=row['neg_log_p'] + 0.15,
        y=row['comparison_clean'],
        text=row['p_label'],
        showarrow=False,
        font=dict(size=11, color='#374151'),
        xanchor='left'
    )

# Add the significance threshold line (primary visual anchor)
threshold = -np.log10(0.05)
fig11.add_vline(
    x=threshold, 
    line_width=2.5, 
    line_dash="dash", 
    line_color="#ef4444"  # Red
)

# Add threshold annotation
fig11.add_annotation(
    x=threshold,
    y=len(mcnemar_plot) - 0.5,
    text="<b>Significance Threshold</b><br>(p = 0.05)",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1.5,
    arrowcolor="#ef4444",
    ax=50,
    ay=-30,
    font=dict(size=10, color="#ef4444"),
    align='left'
)

# Add custom legend
fig11.add_trace(go.Scatter(
    x=[None], y=[None], mode='markers',
    marker=dict(size=12, color=sig_color, symbol='square'),
    name='Significant (p < 0.05)'
))
fig11.add_trace(go.Scatter(
    x=[None], y=[None], mode='markers',
    marker=dict(size=12, color=nonsig_color, symbol='square'),
    name='Not Significant'
))

fig11.update_layout(
    title=(
        '<b>Statistical Significance: McNemar Tests</b><br>'
        '<sup>Pairwise strategy comparisons | T=0.0 | Bars crossing red line are significant</sup>'
    ),
    title_x=0.5,
    template='simple_white',
    height=450, width=950,
    margin=dict(l=200, r=120, t=100, b=70),
    xaxis=dict(
        title='-log₁₀(p-value)  [Longer bar = More significant]',
        range=[0, max(mcnemar_plot['neg_log_p']) + 1.5],
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)'
    ),
    yaxis=dict(title='', categoryorder='array', categoryarray=mcnemar_plot['comparison_clean'].tolist()),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=10)
    ),
    bargap=0.3
)

save_figure(fig11, 'fig09_mcnemar_tests', width=950, height=450)

# =============================================================================
# COMPLETION SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL 9 FIGURES GENERATED SUCCESSFULLY!")
print("="*70)
print(f"""
Generated Figures:
  📊 fig01_model_ranking                    - Horizontal bar chart of model performance
  📊 fig02_strategy_heatmap                 - Heatmap of strategy × model accuracy
  📊 fig03_syntax_vs_nlu_butterfly          - Butterfly chart comparing Syntax vs NLU
  📊 fig04_variant_accuracy                 - Box plot of accuracy by variant type
  📊 fig05_lmarena_correlation              - Scatter plot vs LMArena rank
  📊 fig05_06_benchmark_correlations_combined - Combined LMArena + MMLU correlations
  📊 fig06_mmlu_correlation                 - Scatter plot vs MMLU score
  📊 fig07_belief_bias                      - Congruent vs Incongruent analysis
  📊 fig08_confidence_intervals             - Bootstrap CI visualization
  📊 fig09_mcnemar_tests                    - Statistical significance results

Output Locations:
  📁 Static (PNG/PDF): {FIG_STATIC}
  📁 Interactive (HTML): {FIG_PLOTLY}

Configuration: T={TARGET_TEMP}, Strategy={TARGET_STRATEGY}
Template: plotly_white (built-in)
Color scales: Viridis, Plasma, Turbo, Set2, Pastel, Safe (built-in)
""")
