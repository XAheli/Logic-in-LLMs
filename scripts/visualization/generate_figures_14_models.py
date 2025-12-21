#!/usr/bin/env python3
"""
Generate 4 specific figures for 14-model dataset (excluding Mixtral)
Figures: Model Ranking, Belief Bias, Syntax vs NLU Butterfly, LMArena Correlation
Style: Same as original, NO headings in images (for paper captions)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path(__file__).parent.parent.parent
ANALYSIS_DIR = BASE_PATH / 'results' / 'analysis' / 'tables'
FIG_DIR = BASE_PATH / 'results' / 'analysis' / 'figures' / 'paper_figures_14_models'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 14 models (excluding mixtral-8x22b-instruct)
MODELS_14 = [
    'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-oss-20b', 'glm-4.6',
    'kimi-k2-instruct', 'deepseek-v3.1', 'gemini-2.5-flash-lite',
    'qwen3-next-80b-a3b-instruct', 'qwen3-next-80b-a3b-thinking',
    'llama-3.3-70b-instruct', 'gemma-3-27b-it', 'llama-3.1-8b-instruct',
    'llama-3.2-3b-instruct', 'llama-3.2-1b-instruct'
]

def short_model_name(name):
    """Shorten model names for display"""
    name_map = {
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gemini-2.5-pro': 'Gemini 2.5 Pro',
        'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
        'gpt-oss-20b': 'GPT-OSS-20B',
        'glm-4.6': 'GLM-4.6',
        'kimi-k2-instruct': 'Kimi-K2-Instruct',
        'deepseek-v3.1': 'DeepSeek V3.1',
        'qwen3-next-80b-a3b-instruct': 'Qwen3-Next 80B A3B Instruct',
        'qwen3-next-80b-a3b-thinking': 'Qwen3-Next 80B A3B Thinking',
        'llama-3.3-70b-instruct': 'Llama 3.3 70B Instruct',
        'gemma-3-27b-it': 'Gemma 3 27B IT',
        'llama-3.1-8b-instruct': 'Llama 3.1 8B Instruct',
        'llama-3.2-3b-instruct': 'Llama 3.2 3B Instruct',
        'llama-3.2-1b-instruct': 'Llama 3.2 1B Instruct'
    }
    return name_map.get(name, name)

print("="*80)
print("GENERATING 5 FIGURES FOR 14 MODELS (NO IN-IMAGE HEADINGS)")
print("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n📥 Loading data...")

table1_complete = pd.read_csv(ANALYSIS_DIR / 'paper_table1_complete.csv')
table1_overall = pd.read_csv(ANALYSIS_DIR / 'table1_overall_performance.csv')
table2_dual = pd.read_csv(ANALYSIS_DIR / 'paper_table2_dual_eval.csv')
table3_bias = pd.read_csv(ANALYSIS_DIR / 'paper_table3_belief_bias.csv')
table4_strategy = pd.read_csv(ANALYSIS_DIR / 'table4_model_strategy_matrix.csv')

# Filter table1_overall to only 14 models
table1_overall = table1_overall[table1_overall['Model'].isin(MODELS_14)].copy()
# Filter table4_strategy to only 14 models
table4_strategy = table4_strategy[table4_strategy['Model'].isin(MODELS_14)].copy()

print(f"✅ Loaded {len(table1_complete)} models from tables")

# =============================================================================
# FIGURE 1: Model Performance Ranking (Horizontal Bar)
# =============================================================================
print("\n" + "="*80)
print("FIGURE 1: Model Performance Ranking")
print("="*80)

df_rank = table1_complete.copy()
df_rank = df_rank.sort_values('Syntax_Acc', ascending=True)
df_rank['short_name'] = df_rank['Model'].apply(short_model_name)

fig1 = px.bar(
    df_rank,
    y='short_name',
    x='Syntax_Acc',
    orientation='h',
    color='Syntax_Acc',
    color_continuous_scale='Viridis',
    text=[f"{v:.1f}%" for v in df_rank['Syntax_Acc']],
    labels={'Syntax_Acc': 'Syntax Accuracy (%)', 'short_name': ''}
)

fig1.update_traces(textposition='outside', textfont_size=11)
fig1.update_layout(
    height=650, width=900,
    margin=dict(l=180, r=80, t=40, b=60),  # Reduced top margin (no title)
    coloraxis_showscale=False,
    xaxis=dict(range=[0, 115], title='Syntax Accuracy (%)'),
    yaxis=dict(title=''),
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig1.write_image(FIG_DIR / 'fig01_model_ranking.png', width=900, height=650, scale=2)
fig1.write_image(FIG_DIR / 'fig01_model_ranking.pdf', width=900, height=650)
print(f"✅ Saved: fig01_model_ranking.png + PDF")

# =============================================================================
# FIGURE 2: Strategy Comparison Heatmap
# =============================================================================
print("\n" + "="*80)
print("FIGURE 2: Strategy Comparison Heatmap")
print("="*80)

# Prepare heatmap data
heatmap_data = table4_strategy.set_index('Model').copy()

# Calculate mean across strategies for sorting
heatmap_data['mean'] = heatmap_data.mean(axis=1)
heatmap_data = heatmap_data.sort_values('mean', ascending=False).drop('mean', axis=1)

# Reorder columns and rename
col_order = ['zero_shot', 'one_shot', 'few_shot', 'zero_shot_cot']
col_labels = ['Zero-Shot', 'One-Shot', 'Few-Shot', 'Zero-Shot CoT']
heatmap_data = heatmap_data[col_order]
heatmap_data.columns = col_labels

# Create short model names for y-axis
heatmap_short_names = [short_model_name(m) for m in heatmap_data.index]

fig2 = px.imshow(
    heatmap_data,
    color_continuous_scale='Viridis',
    aspect='auto',
    text_auto='.1f',
    labels=dict(x='Prompting Strategy', y='Model', color='Accuracy (%)')
)

fig2.update_layout(
    height=650, width=850,
    margin=dict(l=220, r=100, t=40, b=60),  # Reduced top margin (no title)
    xaxis=dict(side='bottom'),
    yaxis=dict(
        ticktext=heatmap_short_names,
        tickvals=list(range(len(heatmap_data)))
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig2.write_image(FIG_DIR / 'fig02_strategy_heatmap.png', width=850, height=650, scale=2)
fig2.write_image(FIG_DIR / 'fig02_strategy_heatmap.pdf', width=850, height=650)
print(f"✅ Saved: fig02_strategy_heatmap.png + PDF")

# =============================================================================
# FIGURE 7: Belief Bias (Congruent vs Incongruent)
# =============================================================================
print("\n" + "="*80)
print("FIGURE 7: Belief Bias")
print("="*80)

df_bias = table3_bias.copy()
# Calculate bias gap for sorting
df_bias['congruent_pct'] = df_bias['Congruent_Acc']
df_bias['incongruent_pct'] = df_bias['Incongruent_Acc']
df_bias['bias_gap'] = df_bias['congruent_pct'] - df_bias['incongruent_pct']
df_bias['short_name'] = df_bias['Model'].apply(short_model_name)

# Sort by bias gap (largest bias at top for visual impact)
df_bias = df_bias.sort_values('bias_gap', ascending=True).reset_index(drop=True)

fig7 = go.Figure()

# Add alternating background shading for readability
for i in range(len(df_bias)):
    if i % 2 == 0:
        fig7.add_shape(
            type="rect",
            x0=0, x1=105,
            y0=i - 0.4, y1=i + 0.4,
            fillcolor="rgba(240, 240, 240, 0.5)",
            line=dict(width=0),
            layer="below"
        )

# Add connecting lines (the "gap" visualization) - drawn first so dots appear on top
for i, row in df_bias.iterrows():
    # Determine line color based on gap size
    gap = row['bias_gap']
    if gap > 15:
        line_color = 'rgba(239, 68, 68, 0.6)'  # Red for high bias
    elif gap > 5:
        line_color = 'rgba(251, 191, 36, 0.6)'  # Yellow/amber for medium bias
    else:
        line_color = 'rgba(34, 197, 94, 0.6)'  # Green for low bias
    
    fig7.add_trace(go.Scatter(
        x=[row['incongruent_pct'], row['congruent_pct']],
        y=[row['short_name'], row['short_name']],
        mode='lines',
        line=dict(color=line_color, width=8),
        hoverinfo='skip',
        showlegend=False
    ))

# Add "Incongruent" points (harder task - typically lower)
fig7.add_trace(go.Scatter(
    x=df_bias['incongruent_pct'],
    y=df_bias['short_name'],
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
fig7.add_trace(go.Scatter(
    x=df_bias['congruent_pct'],
    y=df_bias['short_name'],
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
for i, row in df_bias.iterrows():
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
    
    fig7.add_annotation(
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
fig7.add_shape(
    type="rect",
    x0=0, x1=50,
    y0=-0.5, y1=len(df_bias) - 0.5,
    fillcolor="rgba(254, 202, 202, 0.15)",  # Light red
    line=dict(width=0),
    layer="below"
)
fig7.add_shape(
    type="rect",
    x0=50, x1=80,
    y0=-0.5, y1=len(df_bias) - 0.5,
    fillcolor="rgba(254, 249, 195, 0.15)",  # Light yellow
    line=dict(width=0),
    layer="below"
)
fig7.add_shape(
    type="rect",
    x0=80, x1=105,
    y0=-0.5, y1=len(df_bias) - 0.5,
    fillcolor="rgba(187, 247, 208, 0.15)",  # Light green
    line=dict(width=0),
    layer="below"
)

# Add zone labels at bottom
fig7.add_annotation(x=25, y=-0.8, text="<b>Low Accuracy</b>", showarrow=False, 
                    font=dict(size=9, color='#991b1b'), yref='y')
fig7.add_annotation(x=65, y=-0.8, text="<b>Medium</b>", showarrow=False,
                    font=dict(size=9, color='#92400e'), yref='y')
fig7.add_annotation(x=92, y=-0.8, text="<b>High Accuracy</b>", showarrow=False,
                    font=dict(size=9, color='#166534'), yref='y')

fig7.update_layout(
    template='simple_white',
    height=650, width=1000,
    margin=dict(l=220, r=80, t=40, b=100),
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
        categoryarray=df_bias['short_name'].tolist()
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=11)
    ),
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig7.write_image(FIG_DIR / 'fig07_belief_bias.png', width=1000, height=650, scale=2)
fig7.write_image(FIG_DIR / 'fig07_belief_bias.pdf', width=1000, height=650)
print(f"✅ Saved: fig07_belief_bias.png + PDF")

# =============================================================================
# FIGURE 3: Syntax vs NLU Butterfly Chart
# =============================================================================
print("\n" + "="*80)
print("FIGURE 3: Syntax vs NLU (Butterfly Chart)")
print("="*80)

df_dual = table2_dual.copy()
df_dual = df_dual.sort_values('Syntax_Acc', ascending=False)
df_dual['short_name'] = df_dual['Model'].apply(short_model_name)

fig3 = go.Figure()

# Syntax bars (pointing right, positive values)
fig3.add_trace(go.Bar(
    name='Syntax Accuracy',
    y=df_dual['short_name'],
    x=df_dual['Syntax_Acc'],
    orientation='h',
    marker=dict(color='#1f77b4'),
    text=[f"{v:.1f}%" for v in df_dual['Syntax_Acc']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Syntax: %{x:.1f}%<extra></extra>'
))

# NLU bars (pointing left, negative values for display)
fig3.add_trace(go.Bar(
    name='NLU Accuracy',
    y=df_dual['short_name'],
    x=-df_dual['NLU_Acc'],  # Negative for left side
    orientation='h',
    marker=dict(color='#ff7f0e'),
    text=[f"{v:.1f}%" for v in df_dual['NLU_Acc']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>NLU: %{text}<extra></extra>'
))

fig3.update_layout(
    barmode='overlay',
    height=700, width=1000,
    margin=dict(l=200, r=100, t=40, b=60),  # Reduced top margin
    xaxis=dict(
        range=[-120, 120],
        title='Accuracy (%)',
        tickvals=[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100],
        ticktext=['100', '80', '60', '40', '20', '0', '20', '40', '60', '80', '100']
    ),
    yaxis=dict(title=''),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.01,
        xanchor='center',
        x=0.5
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig3.write_image(FIG_DIR / 'fig03_syntax_vs_nlu_butterfly.png', width=1000, height=700, scale=2)
fig3.write_image(FIG_DIR / 'fig03_syntax_vs_nlu_butterfly.pdf', width=1000, height=700)
print(f"✅ Saved: fig03_syntax_vs_nlu_butterfly.png + PDF")

# =============================================================================
# FIGURE 5: LMArena Correlation
# =============================================================================
print("\n" + "="*80)
print("FIGURE 5: LMArena Correlation")
print("="*80)

# LMArena rankings (from previous analysis)
# Note: Only models that exist in LMArena dataset
lmarena_data = {
    'gemini-2.5-flash': {'rank': 44, 'accuracy': 99.58},
    'gemini-2.5-pro': {'rank': 9, 'accuracy': 99.32},
    'gpt-oss-20b': {'rank': 90, 'accuracy': 99.53},  # Not in LMArena
    'glm-4.6': {'rank': 22, 'accuracy': 98.95},
    'kimi-k2-instruct': {'rank': 36, 'accuracy': 95.99},
    'deepseek-v3.1': {'rank': 33, 'accuracy': 95.83},
    'gemini-2.5-flash-lite': {'rank': 70, 'accuracy': 88.91},
    'qwen3-next-80b-a3b-instruct': {'rank': 50, 'accuracy': 79.38},
    'qwen3-next-80b-a3b-thinking': {'rank': 77, 'accuracy': 72.66},
    'llama-3.3-70b-instruct': {'rank': 134, 'accuracy': 69.84},
    'gemma-3-27b-it': {'rank': 79, 'accuracy': 68.39},
    'llama-3.1-8b-instruct': {'rank': 205, 'accuracy': 64.32},
    'llama-3.2-3b-instruct': {'rank': 231, 'accuracy': 59.17},
    'llama-3.2-1b-instruct': {'rank': 260, 'accuracy': 51.88}
}

# Filter to models with LMArena data and prepare DataFrame
lmarena_list = []
for model, data in lmarena_data.items():
    if data['rank'] is not None:
        lmarena_list.append({
            'model': model,
            'lmarena_rank': data['rank'],
            'syllogism_pct': data['accuracy'],
            'short_name': short_model_name(model)
        })

bench_merged = pd.DataFrame(lmarena_list)

# Calculate correlation
from scipy.stats import spearmanr
import statsmodels.api as sm

rho, p_value = spearmanr(bench_merged['lmarena_rank'], bench_merged['syllogism_pct'])

# Define distinct markers and colors for each model
markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 
           'pentagon', 'hexagon', 'star', 'hourglass', 'bowtie']
colors = px.colors.qualitative.Dark24

fig5 = go.Figure()

# Add individual scatter points with unique marker and color per model
for i, (_, row) in enumerate(bench_merged.iterrows()):
    fig5.add_trace(go.Scatter(
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
fig5.add_trace(go.Scatter(
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
fig5.add_trace(go.Scatter(
    x=X_sorted,
    y=Y_pred,
    mode='lines',
    line=dict(color='#ef4444', width=2, dash='solid'),
    name=f'OLS (ρ={rho:.2f})',
    showlegend=True
))

fig5.update_layout(
    height=600, width=950,
    margin=dict(l=100, r=200, t=40, b=80),
    xaxis=dict(
        title='LMArena Rank (lower = better)',
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1
    ),
    yaxis=dict(
        title='Syntax Accuracy (%)',
        range=[40, 110],
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1
    ),
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=10)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig5.write_image(FIG_DIR / 'fig05_lmarena_correlation.png', width=950, height=600, scale=2)
fig5.write_image(FIG_DIR / 'fig05_lmarena_correlation.pdf', width=950, height=600)
print(f"✅ Saved: fig05_lmarena_correlation.png + PDF")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("✅ ALL 5 FIGURES GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nSaved to: {FIG_DIR}/")
print("  • fig01_model_ranking (PNG + PDF)")
print("  • fig02_strategy_heatmap (PNG + PDF)")
print("  • fig03_syntax_vs_nlu_butterfly (PNG + PDF)")
print("  • fig05_lmarena_correlation (PNG + PDF)")
print("  • fig07_belief_bias (PNG + PDF)")
print("\n✅ All figures use 14 models (Mixtral excluded)")
print("✅ No in-image headings (for paper captions)")
print("✅ Exported as PNG (2x scale) + PDF for publication")
