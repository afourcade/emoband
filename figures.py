### SCRIPT FOR GENERATING FIGURES ILLUSTRATING AFFECTIVE DYNAMICS METRICS (MSSD, SD, ICC) AND EMOTIONAL BANDWIDTH

# %% Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.express as px
from pingouin import corr
import statsmodels.stats.multitest as smm
from pathlib import Path

results_dir = Path("D:/EmoBand/results/")
save_path = Path("D:/EmoBand/presentations")

# %% FIGURE 1: MSSD, SD, ICC VISUALIZATION

# Figure 1. Illustration of affective dynamics metrics based on simulated affect ratings. 
# The top row: low versus high mean squared successive difference (MSSD), indexing affective stability versus moment-to-moment instability. 
# The middle row: low versus high standard deviation (SD) indexing affective variability; dashed lines indicate mean affect. 
# The bottom row: low versus high inter-item correlations (ICC), indexing high versus low affective granularity, respectively.

def calculate_mssd(x):
    """Calculate Mean Square Successive Difference"""
    successive_diffs = np.diff(x)
    return np.mean(successive_diffs**2)

def calculate_icc_approx(data):
    """Simplified ICC calculation for visualization purposes"""
    # Between-person variance (variance of means across emotions)
    person_means = data.mean(axis=1)
    between_var = np.var(person_means)
    
    # Within-person variance (average variance within each person)
    within_var = np.mean([np.var(data[i, :]) for i in range(data.shape[0])])
    
    # ICC approximation
    icc = between_var / (between_var + within_var)
    return icc

np.random.seed(42)

# Generate synthetic data
n_timepoints = 50
time = np.arange(n_timepoints)

# === MSSD: Affective Instability ===
# Low MSSD: smooth changes
low_mssd_affect = np.cumsum(np.random.normal(0, 0.3, n_timepoints))
low_mssd_affect = (low_mssd_affect - low_mssd_affect.mean()) / low_mssd_affect.std() * 1.5 + 5

# High MSSD: volatile changes
high_mssd_affect = 5 + np.random.normal(0, 1.5, n_timepoints)
high_mssd_affect[::3] += np.random.choice([-2, 2], size=len(high_mssd_affect[::3]))

# Calculate MSSD
low_mssd_value = calculate_mssd(low_mssd_affect)
high_mssd_value = calculate_mssd(high_mssd_affect)

# === SD: Affect Variability ===
# Low SD - restricted range
low_sd_affect = np.random.normal(5, 0.5, 200)
low_sd_value = np.std(low_sd_affect)

# High SD - wide range
high_sd_affect = np.random.normal(5, 2, 200)
high_sd_value = np.std(high_sd_affect)

# === ICC: Emotional Granularity ===
# Generate emotion ratings
n_obs = 100
emotions = ['Sad', 'Angry', 'Anxious', 'Frustrated', 'Lonely', 'Ashamed']

# Low ICC (high granularity): emotions more independent but still realistic
np.random.seed(42)
# Create base affect with some common variance but preserve independence
common_base = np.random.normal(4, 1.2, n_obs)
low_icc_data = np.zeros((n_obs, len(emotions)))
for i in range(len(emotions)):
    # Each emotion has 60% unique variance, 40% shared
    unique = np.random.normal(0, 1.3, n_obs)
    low_icc_data[:, i] = 0.4 * common_base + 0.6 * unique + np.random.normal(3, 0.5)

# High ICC (low granularity): emotions highly correlated but subtler
np.random.seed(43)
common_factor = np.random.normal(4, 1.5, n_obs)
high_icc_data = np.zeros((n_obs, len(emotions)))
for i in range(len(emotions)):
    # Each emotion has 85% shared variance, 15% unique
    unique = np.random.normal(0, 0.5, n_obs)
    high_icc_data[:, i] = 0.85 * common_factor + 0.15 * unique + np.random.normal(0.5 * i, 0.3)

# Calculate ICC
low_icc_value = calculate_icc_approx(low_icc_data)
high_icc_value = calculate_icc_approx(high_icc_data)

# === CREATE PLOTS ===
fig = plt.figure(figsize=(14, 10))

# 1. MSSD Visualization - Time series
ax1 = plt.subplot(3, 2, 1)
ax1.plot(time, low_mssd_affect, '-', linewidth=2, color='steelblue')
ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Affect Rating', fontsize=11)
ax1.set_title(f'Low MSSD (Stable)\nMSSD = {low_mssd_value:.3f}', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 10)
ax1.grid(alpha=0.3, linewidth=0.5)
ax1.set_xlim(0, 50)

ax2 = plt.subplot(3, 2, 2)
ax2.plot(time, high_mssd_affect, '-', linewidth=2, color='coral')
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Affect Rating', fontsize=11)
ax2.set_title(f'High MSSD (Unstable)\nMSSD = {high_mssd_value:.3f}', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 10)
ax2.grid(alpha=0.3, linewidth=0.5)
ax2.set_xlim(0, 50)

# 2. SD Visualization - Histograms
ax3 = plt.subplot(3, 2, 3)
ax3.hist(low_sd_affect, bins=30, alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
ax3.axvline(np.mean(low_sd_affect), color='black', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(low_sd_affect):.2f}')
ax3.set_xlabel('Affect Rating', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title(f'Low SD (Restricted Range)\nSD = {low_sd_value:.3f}', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, linewidth=0.5)

ax4 = plt.subplot(3, 2, 4)
ax4.hist(high_sd_affect, bins=30, alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
ax4.axvline(np.mean(high_sd_affect), color='black', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(high_sd_affect):.2f}')
ax4.set_xlabel('Affect Rating', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'High SD (Wide Range)\nSD = {high_sd_value:.3f}', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, linewidth=0.5)

# 3. ICC Visualization - Correlation heatmaps
ax5 = plt.subplot(3, 2, 5)
low_icc_corr = np.corrcoef(low_icc_data.T)
im1 = ax5.imshow(low_icc_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax5.set_xticks(range(len(emotions)))
ax5.set_yticks(range(len(emotions)))
ax5.set_xticklabels(emotions, rotation=45, ha='right', fontsize=10)
ax5.set_yticklabels(emotions, fontsize=10)
ax5.set_title(f'Low ICC (High Granularity)\nICC = {low_icc_value:.3f}', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(len(emotions)):
    for j in range(len(emotions)):
        text = ax5.text(j, i, f'{low_icc_corr[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
                       
cbar1 = plt.colorbar(im1, ax=ax5)
cbar1.set_label('Correlation', fontsize=10)

ax6 = plt.subplot(3, 2, 6)
high_icc_corr = np.corrcoef(high_icc_data.T)
im2 = ax6.imshow(high_icc_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax6.set_xticks(range(len(emotions)))
ax6.set_yticks(range(len(emotions)))
ax6.set_xticklabels(emotions, rotation=45, ha='right', fontsize=10)
ax6.set_yticklabels(emotions, fontsize=10)
ax6.set_title(f'High ICC (Low Granularity)\nICC = {high_icc_value:.3f}', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(len(emotions)):
    for j in range(len(emotions)):
        text = ax6.text(j, i, f'{high_icc_corr[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

cbar2 = plt.colorbar(im2, ax=ax6)
cbar2.set_label('Correlation', fontsize=10)

plt.tight_layout()
# remove white background
fig.patch.set_facecolor('none')
plt.savefig(save_path / 'mssd_sd_icc_visualization2.png', dpi=600, bbox_inches='tight')
print("Visualization saved!")

# Print summary statistics
print("\n=== SUMMARY ===")
print(f"\nMSSD (Mean Square Successive Difference):")
print(f"  Low MSSD (stable):   {low_mssd_value:.3f}")
print(f"  High MSSD (unstable): {high_mssd_value:.3f}")
print(f"  Interpretation: Higher values = more volatility/instability")

print(f"\nSD (Standard Deviation):")
print(f"  Low SD (restricted):  {low_sd_value:.3f}")
print(f"  High SD (wide range): {high_sd_value:.3f}")
print(f"  Interpretation: Higher values = greater overall variability")

print(f"\nICC (Intraclass Correlation):")
print(f"  Low ICC (high granularity):  {low_icc_value:.3f}")
print(f"  High ICC (low granularity):  {high_icc_value:.3f}")
print(f"  Interpretation: Lower ICC = emotions are more differentiated")
print(f"                  Higher ICC = emotions move together (low differentiation)")

# %% FIGURE 2: BANDWIDTH

# Figure 2. Three-dimensional state space of negative affect. 
# Here only 3 emotion items are considered, for visualization purposes. 
# H = Shannon’s Entropy measure.

def multivariate_shannon_entropy_from_time_series(*args, n_levels, ranges=None, base=2):
    """
    Calculate multivariate Shannon entropy from multiple continuous time series,
    discretized into n_levels each by histogram binning.
    """
    data = np.array(args).T  # shape (n_samples, n_signals)
    
    # N-D histogram to get joint counts
    joint_counts, edges = np.histogramdd(np.array(data), bins=n_levels, range=ranges)
    
    # Convert counts to proportions (joint probability distribution)
    joint_prob = joint_counts / np.sum(joint_counts)
    
    # Filter out zero probabilities to avoid log(0)
    joint_prob = joint_prob[joint_prob > 0]
    
    # Compute multivariate Shannon entropy
    entropy = -np.sum(joint_prob * np.log(joint_prob) / np.log(base))
    
    return entropy

np.random.seed(42)

# Define negative emotions (using exact words from bandwidth study)
# For visualization: worried, low, irritable + 3 others for 6D calculation
emotions = ['Low', 'Irritable', 'Worried', 'Frustrated', 'Sad', 'Ashamed']
n_emotions = len(emotions)
n_obs = 200

# === Low Bandwidth - limited emotional state space ===
np.random.seed(43)
low_bandwidth_data = np.zeros((n_obs, n_emotions))

# Create 3 distinct emotional states that repeat
state1 = np.array([5, 1, 5, 4, 2, 1])  # Low + Worried + Frustrated
state2 = np.array([4, 1, 6, 5, 1, 1])  # Worried + Frustrated dominant
state3 = np.array([6, 2, 4, 3, 2, 2])  # Low + Worried

# Assign states with some noise
for i in range(n_obs):
    state_choice = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
    if state_choice == 0:
        low_bandwidth_data[i] = state1 + np.random.normal(0, 0.4, n_emotions)
    elif state_choice == 1:
        low_bandwidth_data[i] = state2 + np.random.normal(0, 0.4, n_emotions)
    else:
        low_bandwidth_data[i] = state3 + np.random.normal(0, 0.4, n_emotions)

low_bandwidth_data = np.clip(low_bandwidth_data, 1, 7)

# === High Bandwidth - diverse emotional state space ===
np.random.seed(44)
high_bandwidth_data = np.zeros((n_obs, n_emotions))

# Each observation is a unique combination
for i in range(n_obs):
    high_bandwidth_data[i] = np.random.uniform(1.5, 6.5, n_emotions)
    # Add weak correlation but maintain diversity
    common = np.random.normal(0, 0.7)
    high_bandwidth_data[i] += 0.3 * common

high_bandwidth_data = np.clip(high_bandwidth_data, 1, 7)

# Calculate bandwidth (multivariate Shannon entropy)
n_levels = 5
ranges = [(1, 7)] * n_emotions

low_bandwidth_entropy = multivariate_shannon_entropy_from_time_series(
    *[low_bandwidth_data[:, i] for i in range(n_emotions)],
    n_levels=n_levels,
    ranges=ranges
)

high_bandwidth_entropy = multivariate_shannon_entropy_from_time_series(
    *[high_bandwidth_data[:, i] for i in range(n_emotions)],
    n_levels=n_levels,
    ranges=ranges
)

max_entropy = np.log2(n_levels ** n_emotions)

# Discretize for state counting
low_discrete = np.digitize(low_bandwidth_data, bins=np.linspace(1, 7, n_levels+1)) - 1
unique_states_low, counts_low = np.unique(low_discrete, axis=0, return_counts=True)

high_discrete = np.digitize(high_bandwidth_data, bins=np.linspace(1, 7, n_levels+1)) - 1
unique_states_high, counts_high = np.unique(high_discrete, axis=0, return_counts=True)

# === CREATE 3D HISTOGRAMS ===
# Use Low (0), Irritable (1), Worried (2)
triplet = (0, 1, 2)  # Low, Irritable, Worried
triplet_names = ['Low', 'Irritable', 'Worried']

fig = plt.figure(figsize=(18, 8))

def plot_3d_histogram(data, emotions_subset, emotion_names, ax, title, color, background_color='none'):
    """Create a 3D scatter plot with points colored by frequency"""
    # Extract the three emotions
    x = data[:, emotions_subset[0]]
    y = data[:, emotions_subset[1]]
    z = data[:, emotions_subset[2]]
    
    # Create 3D histogram to get counts
    hist, edges = np.histogramdd(np.array([x, y, z]).T, bins=5, range=[(1, 7), (1, 7), (1, 7)])
    
    # Get bin centers and counts
    xcenters = (edges[0][:-1] + edges[0][1:]) / 2
    ycenters = (edges[1][:-1] + edges[1][1:]) / 2
    zcenters = (edges[2][:-1] + edges[2][1:]) / 2
    
    # Create arrays for plotting
    xpos, ypos, zpos, counts = [], [], [], []
    
    for i in range(len(xcenters)):
        for j in range(len(ycenters)):
            for k in range(len(zcenters)):
                if hist[i, j, k] > 0:  # Only plot non-zero bins
                    xpos.append(xcenters[i])
                    ypos.append(ycenters[j])
                    zpos.append(zcenters[k])
                    counts.append(hist[i, j, k])
    
    if len(xpos) > 0:
        # Normalize counts for color mapping
        counts = np.array(counts)
        # Size proportional to count
        sizes = (counts / counts.max()) * 500 + 50
        
        # Create scatter plot
        scatter = ax.scatter(xpos, ypos, zpos, c=counts, cmap=color, 
                           s=sizes, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Frequency', fontsize=11)
    
    ax.set_xlabel(emotion_names[0], fontsize=13, labelpad=12, fontweight='bold')
    ax.set_ylabel(emotion_names[1], fontsize=13, labelpad=12, fontweight='bold')
    ax.set_zlabel(emotion_names[2], fontsize=13, labelpad=12, fontweight='bold')
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_zlim(1, 7)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_zticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=30)
    ax.view_init(elev=25, azim=50)
    ax.grid(True, alpha=0.3)
    # Set background color
    ax.set_facecolor(background_color)
    # set pane colors to white with no transparency
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))




# Low bandwidth
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_3d_histogram(low_bandwidth_data, triplet, triplet_names, ax1, 
                  f'Low Bandwidth\nH = {low_bandwidth_entropy:.2f} bits ({unique_states_low.shape[0]} unique states)',
                  'Blues', background_color='none')


# High bandwidth
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_3d_histogram(high_bandwidth_data, triplet, triplet_names, ax2,
                  f'High Bandwidth\nH = {high_bandwidth_entropy:.2f} bits ({unique_states_high.shape[0]} unique states)',
                  'Reds', background_color='none')

plt.suptitle('Three-Dimensional State Space of Negative Affect', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.1)
# remove white background
fig.patch.set_facecolor('none')
plt.savefig(save_path / 'bandwidth_3d_simple2.png', dpi=600, bbox_inches='tight')
print("Visualization saved!")

print("\n=== BANDWIDTH SUMMARY ===")
print(f"\nLow Bandwidth:")
print(f"  Shannon Entropy: {low_bandwidth_entropy:.2f} bits")
print(f"  Unique 6D states: {len(unique_states_low)}")
print(f"  Visual: Bars concentrated in few regions (clustered)")

print(f"\nHigh Bandwidth:")
print(f"  Shannon Entropy: {high_bandwidth_entropy:.2f} bits")
print(f"  Unique 6D states: {len(unique_states_high)}")
print(f"  Visual: Bars dispersed across space (diverse)")

print(f"\nNote: Visualization shows Low-Irritable-Worried triplet")
print(f"      but entropy is calculated on all 6 emotions")

# %% FIGURE 3: CORRELATION HEATMAP

#Figure 3. Bivariate correlations between all pairs of variables of interest. 
# Only significant correlations (pFDR < .05) are colored.

# Load the wide format data used in analysis
emo_band_data_wide = pd.read_csv(results_dir / "emo_band_granularity_wide.tsv", sep='\t')

# Define variables and rename dictionary
variables_of_interest = ['gender', 'age', 'rucla_kw', 'cesd_kw', 'bandwidth_positive', 'granularity_positive', 'sd_positive', 'wmssd_positive', 'inertia_positive', 'mean_affect_level_positive', 'bandwidth_negative', 'granularity_negative', 'sd_negative', 'wmssd_negative', 'inertia_negative', 'mean_affect_level_negative']

rename_dict = {
    'rucla_kw': 'Loneliness',
    'cesd_kw': 'Depressive Symptoms',
    'granularity_positive': 'Granularity (POS)',
    'granularity_negative': 'Granularity (NEG)',
    'bandwidth_positive': 'Bandwidth (POS)',
    'bandwidth_negative': 'Bandwidth (NEG)',
    'age': 'Age',
    'gender': 'Gender (0=F, 1=M)',
    'mean_affect_level_positive': 'Mean Affect Level (POS)',
    'mean_affect_level_negative': 'Mean Affect Level (NEG)',
    'wmssd_positive': 'Instability (POS)',
    'wmssd_negative': 'Instability (NEG)',
    'sd_positive': 'Variability (POS)',
    'sd_negative': 'Variability (NEG)',
    'inertia_positive': 'Inertia (POS)',
    'inertia_negative': 'Inertia (NEG)'
} 

# Select variables of interest without gender, age, inertia
corr_vars = variables_of_interest.copy()
corr_vars.remove('gender')
corr_vars.remove('age')
corr_vars.remove('inertia_positive')
corr_vars.remove('inertia_negative')
corr_matrix = emo_band_data_wide[corr_vars].copy()

# Create a correlation table
corr_table = corr_matrix[corr_vars].corr().round(3)
corr_table.rename(index=rename_dict, columns=rename_dict, inplace=True)

# Create p-value matrix
pvals = np.zeros((len(corr_vars), len(corr_vars)))
for i, var1 in enumerate(corr_vars):
    for j, var2 in enumerate(corr_vars):
        if i != j:
            res = corr(corr_matrix[var1], corr_matrix[var2])
            pvals[i, j] = res['p-val'].values[0]
        else:
            pvals[i, j] = 0

# Correct p-values for multiple comparisons using False Discovery Rate (FDR) correction
pvals_flat = pvals[np.triu_indices_from(pvals, k=1)]
_, pvals_corrected, _, _ = smm.multipletests(pvals_flat, alpha=0.05, method='fdr_bh')
pvals_corrected_matrix = np.zeros_like(pvals)
pvals_corrected_matrix[np.triu_indices_from(pvals, k=1)] = pvals_corrected
pvals_corrected_matrix += pvals_corrected_matrix.T

# Create mask for non-significant correlations
mask = pvals_corrected_matrix > 0.05

# Create figure - show only lower triangle
corr_table_lower = corr_table.copy()
# Mask upper triangle
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        if j > i:
            corr_table_lower.iloc[i, j] = np.nan

fig3 = px.imshow(corr_table_lower, text_auto=True, aspect="equal",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0.0,
                range_color=[-1, 1],
                labels=dict(color="Correlation"),
                x=corr_table_lower.columns,
                y=corr_table_lower.index,
                height=1200,
                width=1200)
fig3.update_coloraxes(colorbar=dict(len=0.7, thickness=30))

# Reduce opacity of colors if p-value > 0.05 by overlaying semi-transparent white rectangles
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        if j <= i and mask[i, j]:  # only apply to lower triangle
            # Add a semi-transparent white rectangle over non-significant cells
            fig3.add_shape(
                type="rect",
                x0=j-0.5, x1=j+0.5,
                y0=i-0.5, y1=i+0.5,
                fillcolor="white",
                opacity=1,
                layer="above",
                line_width=0
            )
            # Add the correlation value as text annotation in black
            corr_value = corr_table_lower.iloc[i, j]
            fig3.add_annotation(
                x=j, y=i,
                text=f"{corr_value:.3f}",
                showarrow=False,
                font=dict(color="black", size=12),
                xref="x", yref="y"
            )

# Make background transparent and remove grid
fig3.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=False)
fig3.show()
# set font size of axis labels and ticks
fig3.update_xaxes(tickfont=dict(size=16), title_font=dict(size=16, weight='bold'))
fig3.update_yaxes(tickfont=dict(size=16), title_font=dict(size=16, weight='bold'))

# Save figure
#fig3_filename = results_dir / 'emo_band_granularity_correlation_heatmap.html'
#fig3.write_html(fig3_filename)
fig3_png_filename = save_path / 'emo_band_granularity_correlation_heatmap.png'
fig3.write_image(fig3_png_filename, width=1200, height=1200, scale=7)

print("\nCorrelation heatmap saved!")

# %%
