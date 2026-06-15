"""
Script to compute the bandwidth of positive and negative emotion ratings in EmoBand dataset.
Emotion Bandwidth: the number of unique rating values used by each rater.

The following steps are performed:
1. Load and preprocess the data (compute emotion bandwidth, granularity, mean affect level,
instability, variability for each participant and affective dimension).
2. Compute correlations between emotion bandwidth, granularity, instability, variability, mean affect level,
and questionnaire scores (loneliness, depression).
3. Run regression analyses to test whether emotion bandwidth, granularity, instability, variability predict loneliness
and depression, controlling for mean affect level, age, and gender

Author: Aleksandra Piejka, Antonin Fourcade (piejka[at]cbs.mpg.de, antonin.fourcade[at]maxplanckschools.de)
Last version: 15.06.2026
"""

# %% IMPORTS AND SETTINGS
# import packages
from cProfile import label
from operator import index
from pathlib import Path
import pandas as pd
import numpy as np
from pingouin import corr
import plotly.express as px
import statsmodels.formula.api as smf
import pingouin as pg
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multitest as smm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# Set paths and experiment parameters
exp_name = "emo_band"
exp_dir = "D:/EmoBand/" # to change to your local path where the data and results folders are located
results_dir =  exp_dir + "results/"
data_dir = exp_dir + "data/"
data_path = data_dir + "esm_clean_trimmed.csv"
demographic_path = data_dir + "demographic_data.csv"
save_data_filename = 'emo_band_proc_data'
save_corr_fig_filename = 'emo_band_correlation'
save_corr_table_filename = 'emo_band_correlation_matrix'
save_desc_stats_filename = 'emo_band_descriptive_stats'
save_reg_filename = 'emo_band_regression_results'
save_med_filename = 'emo_band_mediation_results'
save_type = "tsv"  # save type for data loss results, can be "json" or "tsv"
sep = '\t' if save_type == 'tsv' else ','

# number of levels for granularity (levels in Likert scale)
n_levels = 7

# affective dimensions of interest
affect_dimensions = {
    'positive': ['pa_joyful_rand', 'pa_cheerful_rand', 'pa_happy_rand', 'pa_content_rand', 'pa_relaxed_rand', 'pa_energetic_rand'],
    'negative': ['na_tense_rand', 'na_irritable_rand', 'na_worried_rand', 'na_low_rand', 'na_lonely_rand', 'na_abandoned_rand']
}

# standardize variables of interest before analyses
standardize = True 

# %% FUNCTIONS
# univariate shannon entropy
def shannon_entropy_from_time_series(x, n_levels, range_x=(-1, 1), base=2):
    """
    Calculate Shannon entropy from a time series which is discretized into levels.

    Parameters:
        x (list or np.array): Time series.
        n_levels (int): Number of discrete scale levels for discretization.
        range_x (tuple): Range of the scale levels (default (-1, 1)).
        base (int): Logarithm base (default 2 for bits).

    Returns:
        float: Shannon entropy based on proportion of time spent at each level.
    """
    x = np.array(x)
    # histogram
    counts, _ = np.histogram(x, bins=n_levels, range=range_x)
    # Convert counts to proportions
    proportions = counts / counts.sum()
    # Remove zero proportions
    proportions = proportions[proportions > 0]
    entropy = -np.sum(proportions * np.log(proportions) / np.log(base))
    return entropy

# multivariate shannon entropy
def multivariate_shannon_entropy_from_time_series(*args, n_levels, ranges=None, base=2):
    """
    Calculate multivariate Shannon entropy from multiple continuous time series,
    discretized into n_levels each by histogram binning.

    Parameters:
        *args: Multiple time series (each as a list or np.array).
        n_levels (int): Number of discrete levels for discretization (same for all signals).
        ranges (list of tuples): List of tuples specifying the (min, max) range for each time series.
        base (int): Logarithm base (default 2 for bits).

    Returns:
        float: Multivariate Shannon entropy based on joint distribution of discretized levels.
    """
    data = np.array(args).T  # shape (n_samples, n_signals)
    assert all(len(data[0]) == len(d) for d in data[1:]), "All time series must have the same length."

    # N-D histogram to get joint counts
    joint_counts, edges = np.histogramdd(np.array(data), bins=n_levels, range=ranges)

    # Convert counts to proportions (joint probability distribution)
    joint_prob = joint_counts / np.sum(joint_counts)

    # Filter out zero probabilities to avoid log(0)
    joint_prob = joint_prob[joint_prob > 0]

    # Compute bivariate Shannon entropy
    entropy = -np.sum(joint_prob * np.log(joint_prob) / np.log(base))

    return entropy


def icc_pingouin(data, emotion_cols):
    """
    Calculate ICC using pingouin package.

    Parameters:
        data (pd.DataFrame): DataFrame with emotion ratings.
        emotion_cols (list): List of columns corresponding to emotions.
    
    Returns:
        float: ICC value.
    """
    # melt data to long format
    data_long = data.melt(id_vars=['participant'], value_vars=emotion_cols, var_name='emotion', value_name='rating')
    data_long['occasion'] = np.tile(np.arange(1, len(data_long) // len(emotion_cols) + 1), len(emotion_cols))
    icc_result = pg.intraclass_corr(data=data_long, targets='occasion', raters='emotion', ratings='rating')
    icc_value = icc_result[icc_result['Type'] == 'ICC3k']['ICC'].values[0]
    return icc_value

def emotion_granularity(icc):
    """
    Convert ICC to emotion granularity measure.
    Emotion granularity = 1 - ICC

    Parameters:
        icc (float): Intraclass correlation coefficient.

    Returns:
        float: Emotion granularity.
    """
    return 1 - icc

def compute_weighted_mssd(sub_data, affect_dimensions, time_col='timestamp_response'):
    """
    Compute time-interval-weighted MSSD for each affect item, then average 
    within affective dimensions (e.g., positive, negative).
    
    Parameters:
        sub_data (pd.DataFrame): DataFrame with data for a single subject.
        affect_dimensions (dict): Dict mapping dimension names to lists of columns.
        time_col (str): Column name for timestamps.
    
    Returns:
        dict: Flat dictionary with formatted column names.
    """
    sub_data = sub_data.sort_values(time_col).copy()
    
    timestamps = pd.to_datetime(sub_data[time_col])
    time_diffs = timestamps.diff().dt.total_seconds().values[1:]  # in seconds
    
    time_diffs = np.where(time_diffs <= 0, np.nan, time_diffs)
    
    results = {}
    
    for dimension, cols in affect_dimensions.items():
        suffix = 'positive' if dimension == 'positive' else 'negative'
        item_wmssd = []
        
        for col in cols:
            values = sub_data[col].values
            
            valid_mask = ~np.isnan(values[:-1]) & ~np.isnan(values[1:]) & ~np.isnan(time_diffs)
            
            if valid_mask.sum() < 1:
                wmssd = np.nan
            else:
                diffs = np.diff(values)[valid_mask]
                intervals = time_diffs[valid_mask]
                weights = 1.0 / intervals
                squared_diffs = diffs ** 2
                wmssd = np.sum(weights * squared_diffs) / np.sum(weights)
            
            # Format column name
            item_name = col.replace('_rand', '_wmssd')
            results[f'{item_name}'] = wmssd
            item_wmssd.append(wmssd)
        
        valid_wmssd = [v for v in item_wmssd if not np.isnan(v)]
        results[f'wmssd_{suffix}'] = np.mean(valid_wmssd) if valid_wmssd else np.nan
    
    return results

def compute_variability(sub_data, affect_dimensions):
    """
    Compute standard deviation for each affect item, then average 
    within affective dimensions (e.g., positive, negative).
    
    Parameters:
        sub_data (pd.DataFrame): DataFrame with data for a single subject.
        affect_dimensions (dict): Dict mapping dimension names to lists of columns.
    Returns:
        dict: Flat dictionary with formatted column names.
    """    
    results = {}
    
    for dimension, cols in affect_dimensions.items():
        suffix = 'positive' if dimension == 'positive' else 'negative'
        item_sd = []
        
        for col in cols:
            values = sub_data[col].values
            sd = np.nanstd(values)
            
            # Format column name
            item_name = col.replace('_rand', '_sd')
            results[f'{item_name}'] = sd
            item_sd.append(sd)
        
        valid_sd = [v for v in item_sd if not np.isnan(v)]
        results[f'sd_{suffix}'] = np.mean(valid_sd) if valid_sd else np.nan
    
    return results


def compute_inertia_simple_adjusted(sub_data, affect_dimensions):
    """Lag-1 AR controlling for time gap (Δt)."""
    results = {}
    sub_data = sub_data.sort_values("timestamp_response").reset_index(drop=True)
    sub_data['dt'] = sub_data['timestamp_response'].diff() 
    
    for dimension, cols in affect_dimensions.items():
        suffix = 'positive' if dimension == 'positive' else 'negative'
        item_inertia = []
        
        for col in cols:
            if col not in sub_data.columns:
                continue
            df = sub_data[[col, 'dt']].dropna()
            if len(df) < 4:
                inertia = np.nan
            else:
                import statsmodels.api as sm
                df['lag'] = df[col].shift(1)
                df = df.dropna()
                X = sm.add_constant(df[['lag', 'dt']]) # add intercept
                model = sm.OLS(df[col], X).fit()
                inertia = model.params['lag']  # AR coefficient
            
            # Format column name
            item_name = col.replace('_rand', '_inertia')
            results[f'{item_name}'] = inertia
            item_inertia.append(inertia)
            results[f"inertia_{suffix}_{col}"] = inertia
        
        valid_inertia = [v for v in item_inertia if not np.isnan(v)]
        results[f"inertia_{suffix}_mean"] = float(np.nanmean(valid_inertia)) if valid_inertia else np.nan
    return results

##--------------------------------------------------------------------------
## %% PREPROCESSING 
##--------------------------------------------------------------------------

# %% LOAD DATA
# load data
data = pd.read_csv(data_path)

# %% ALL SUBJECTS
# initialize results collection
rows = []
# loop over subjects
subject_ids = data['participant'].unique()
for sub_id in subject_ids:
    sub_data = data[data['participant'] == sub_id]

    # compute multivariate entropy for each affect dimension
    for aff_dim, dimensions in affect_dimensions.items():
        data_aff = sub_data[dimensions]
        entropy_mv = multivariate_shannon_entropy_from_time_series(*[data_aff[col].values for col in dimensions],
                                                                    n_levels=n_levels,
                                                                    ranges=[(1, 7)]*len(dimensions),
                                                                    base=2)
        
        # calculate the number of states possible with given dimensions and levels
        num_states = n_levels ** len(dimensions)
        num_prompts = len(sub_data)
        if num_prompts < num_states:
            num_possible_states = num_prompts
        else:
            num_possible_states = num_states
        # compute emotion bandwidth (normalized by number of possible states)
        emo_bw_mv = 2**entropy_mv / num_possible_states
        #emo_bw_mv = np.round(emo_bw_mv, 1)
        # compute mean affect level across all dimensions and prompts
        mean_level = data_aff.mean().mean()
        # compute ICC and granularity
        icc_val = icc_pingouin(sub_data, dimensions)
        granularity = emotion_granularity(icc_val)

        # if granularity > 1, print warning
        if granularity > 1:
            print(f'Warning: Emotion granularity > 1 for subject {sub_id}, affective dimension {aff_dim}: {icc_val}')

        # store results
        row = {
            'participant': sub_id,
            'affective_dimension': aff_dim,
            'multivariate_entropy': entropy_mv,
            'bandwidth': emo_bw_mv,
            'mean_affect_level': mean_level,
            'icc': icc_val,
            'granularity': granularity, 
        }
        rows.append(row)

# build DataFrame
emo_band_data = pd.DataFrame(rows)

# delete both rows for participants with emotion granularity (either negative or positive) bigger than 1
emo_band_data = emo_band_data[~emo_band_data['participant'].isin(
    emo_band_data[emo_band_data['granularity'] > 1]['participant'].unique()
)]
#%% ADD INSTABILITY (WEIGHTED MSSD)

wmssd_rows = []
for sub_id in subject_ids:
    sub_data = data[data['participant'] == sub_id]
    
    # positive affect - pooled across days
    results = compute_weighted_mssd(sub_data, affect_dimensions, time_col='timestamp_response')
    wmssd_rows.append({
        'participant': sub_id,
        'pa_joyful_instability': results['pa_joyful_wmssd'],
        'pa_cheerful_instability': results['pa_cheerful_wmssd'],
        'pa_happy_instability':  results['pa_happy_wmssd'],
        'pa_content_instability': results['pa_content_wmssd'],
        'pa_relaxed_instability': results['pa_relaxed_wmssd'],
        'pa_energetic_instability': results['pa_energetic_wmssd'],
        'instability_positive': results['wmssd_positive'],  # mean of above 6
        'na_tense_instability': results['na_tense_wmssd'],
        'na_irritable_instability': results['na_irritable_wmssd'],
        'na_worried_instability': results['na_worried_wmssd'],
        'na_low_instability': results['na_low_wmssd'],
        'na_lonely_instability': results['na_lonely_wmssd'],
        'na_abandoned_instability': results['na_abandoned_wmssd'],
        'instability_negative': results['wmssd_negative']  # mean of above 6
})

wmssd_data = pd.DataFrame(wmssd_rows)

# merge with emo_band_data
emo_band_data = emo_band_data.merge(wmssd_data, on='participant', how='left')

# %% ADD VARIABILITY (SD)   
variability_rows = []
for sub_id in subject_ids:
    sub_data = data[data['participant'] == sub_id]
    
    # positive affect - pooled across days
    results = compute_variability(sub_data, affect_dimensions)
    variability_rows.append({
        'participant': sub_id,
        'pa_joyful_variability': results['pa_joyful_sd'],
        'pa_cheerful_variability': results['pa_cheerful_sd'],
        'pa_happy_variability':  results['pa_happy_sd'],
        'pa_content_variability': results['pa_content_sd'],
        'pa_relaxed_variability': results['pa_relaxed_sd'],
        'pa_energetic_variability': results['pa_energetic_sd'],
        'variability_positive': results['sd_positive'],  # mean of above 6
        'na_tense_variability': results['na_tense_sd'],
        'na_irritable_variability': results['na_irritable_sd'],
        'na_worried_variability': results['na_worried_sd'],
        'na_low_variability': results['na_low_sd'],
        'na_lonely_variability': results['na_lonely_sd'],
        'na_abandoned_variability': results['na_abandoned_sd'],
        'variability_negative': results['sd_negative']  # mean of above 6
})
variability_data = pd.DataFrame(variability_rows)
# merge with emo_band_data
emo_band_data = emo_band_data.merge(variability_data, on='participant', how='left')

# %% ADD QUESTIONNAIRE SCORES
# add columns cols from data to emo_band_data
cols = ['RUCLA_kw', 'CESD_kw'] # RUCLA loneliness and CESD depression scores
for col in cols:
    col_data = data[['participant', col]].drop_duplicates()
    emo_band_data = emo_band_data.merge(col_data, on='participant', how='left')
# remove '_kw' suffix from column names
emo_band_data = emo_band_data.rename(columns={col: col.replace('_kw', '') for col in cols})
# decapitalize column names
emo_band_data.columns = emo_band_data.columns.str.lower()

# %% ADD DEMOGRAPHIC INFO
# load demographic data from file
demographic_data = pd.read_csv(demographic_path)
emo_band_data = emo_band_data.merge(demographic_data, on='participant', how='left')

# %% SAVE emo_band_data
# create results directory if it doesn't exist
Path(results_dir).mkdir(parents=True, exist_ok=True)
sep = '\t' if save_type == 'tsv' else ','
emo_band_data_filename = results_dir + '/' + save_data_filename + '.' + save_type
# save emo_band_data
emo_band_data.to_csv(emo_band_data_filename, index=False, sep=sep)

# %% PIVOT DATA TO WIDE FORMAT
# pivot data from long to wide format
emo_band_data_wide = emo_band_data.pivot(index=['participant', 'rucla', 'gender', 'cesd', 'age', 'instability_positive', 'instability_negative', 'variability_positive', 'variability_negative'], columns=['affective_dimension'], values=['granularity', 'bandwidth', 'mean_affect_level'])
# rename columns without emotion_ prefix
emo_band_data_wide.columns = ['_'.join(col).strip() for col in emo_band_data_wide.columns.values]
emo_band_data_wide.columns = emo_band_data_wide.columns.str.replace('emotion_', '')
# reset index
emo_band_data_wide = emo_band_data_wide.reset_index()
# convert gender to numeric
emo_band_data_wide['gender'] = emo_band_data_wide['gender'].map({'F': 0, 'M': 1})
# convert other columns to numeric
cols_to_numeric = ['granularity_positive', 'granularity_negative', 'bandwidth_positive', 'bandwidth_negative', 'rucla', 'cesd', 'age', 'instability_positive', 'instability_negative', 'variability_positive', 'variability_negative']
emo_band_data_wide[cols_to_numeric] = emo_band_data_wide[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

# save emo_band_data_wide
emo_band_data_wide_filename = results_dir + '/' + save_data_filename + '_wide.' + save_type
emo_band_data_wide.to_csv(emo_band_data_wide_filename, index=False, sep=sep)

# remove rows with NaN values in key columns
emo_band_data_wide = emo_band_data_wide.dropna(subset=['granularity_positive', 'granularity_negative', 'bandwidth_positive', 'bandwidth_negative', 'rucla', 'cesd'])

print('Number of participants: ', len(emo_band_data_wide))

##--------------------------------------------------------------------------
## %% ANALYSES
##--------------------------------------------------------------------------

# %% LOAD DATA
# load emo_band_data_wide
emo_band_data_wide_filename = results_dir + '/' + save_data_filename + '_wide.' + save_type
emo_band_data_wide = pd.read_csv(emo_band_data_wide_filename, sep=sep)

# %% SELECTION OF VARIABLES OF INTEREST
variables_of_interest = ['gender', 'age', 'rucla', 'cesd', 'bandwidth_positive', 'granularity_positive', 'variability_positive', 'instability_positive', 'mean_affect_level_positive', 'bandwidth_negative', 'granularity_negative', 'variability_negative', 'instability_negative', 'mean_affect_level_negative']
# rename variables for better readability
rename_dict = {
    'rucla': 'Loneliness',
    'cesd': 'Depressive Symptoms',
    'granularity_positive': 'Granularity (POS)',
    'granularity_negative': 'Granularity (NEG)',
    'bandwidth_positive': 'Bandwidth (POS)',
    'bandwidth_negative': 'Bandwidth (NEG)',
    'age': 'Age',
    'gender': 'Gender (0=F, 1=M)',
    'mean_affect_level_positive': 'Mean Affect Level (POS)',
    'mean_affect_level_negative': 'Mean Affect Level (NEG)',
    'instability_positive': 'Instability (POS)',
    'instability_negative': 'Instability (NEG)',
    'variability_positive': 'Variability (POS)',
    'variability_negative': 'Variability (NEG)'
}   
# %% CORRELATION
# select variables of interest without gender and age
corr_vars = variables_of_interest.copy()
corr_vars.remove('gender')
corr_vars.remove('age')
corr_matrix = emo_band_data_wide[corr_vars].copy()

# Create a correlation table
corr_table = corr_matrix[corr_vars].corr().round(3)
corr_table.rename(index=rename_dict, columns=rename_dict, inplace=True)

# create p-value matrix
pvals = np.zeros((len(corr_vars), len(corr_vars)))
for i, var1 in enumerate(corr_vars):
    for j, var2 in enumerate(corr_vars):
        if i != j:
            res = corr(corr_matrix[var1], corr_matrix[var2])
            pvals[i, j] = res['p-val'].values[0]
        else:
            pvals[i, j] = 0
# correct p-values for multiple comparisons using False Discovery Rate (FDR) correction
pvals_flat = pvals[np.triu_indices_from(pvals, k=1)]
_, pvals_corrected, _, _ = smm.multipletests(pvals_flat, alpha=0.05, method='fdr_bh')
pvals_corrected_matrix = np.zeros_like(pvals)
pvals_corrected_matrix[np.triu_indices_from(pvals, k=1)] = pvals_corrected
pvals_corrected_matrix += pvals_corrected_matrix.T

# create mask for non-significant correlations
mask = pvals_corrected_matrix > 0.05

# create figure with only lower triangle
corr_table_lower = corr_table.copy()
# Mask upper triangle
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        if j > i:
            corr_table_lower.iloc[i, j] = np.nan
# use plotly to create heatmap of correlation table with masked upper triangle and p-value-based opacity
fig = px.imshow(corr_table_lower, text_auto=True, aspect="equal",
                color_continuous_scale='RdBu',
                color_continuous_midpoint = 0.0,
                range_color=[-1, 1],
                labels=dict(color="Correlation"),
                x=corr_table_lower.columns,
                y=corr_table_lower.index,
                height=1200,
                width=1200)
fig.update_coloraxes(colorbar=dict(len=0.7, thickness=30))
# reduce opacity of colors if p-value > 0.05 by overlaying semi-transparent white rectangles
for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        if j <= i and mask[i, j]:  # only apply to lower triangle
            # Add a semi-transparent white rectangle over non-significant cells
            fig.add_shape(
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
            fig.add_annotation(
                x=j, y=i,
                text=f"{corr_value:.3f}",
                showarrow=False,
                font=dict(color="black", size=12),
                xref="x", yref="y"
            )

# Make background transparent and remove grid
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# save figure as png
fig_png_filename = results_dir + '/' + save_corr_fig_filename + '.png'
fig.write_image(fig_png_filename, width=1200, height=1200, scale=2)

# save correlation table
corr_table_filename = results_dir + '/' + save_corr_table_filename + '.' + save_type
corr_table.to_csv(corr_table_filename, sep=sep)

# %% DESCRIPTIVE STATISTICS
desc_stats = emo_band_data_wide[variables_of_interest].describe().T

desc_stats.rename(index=rename_dict, inplace=True)

# save descriptive statistics
desc_stats_filename = results_dir + '/' + save_desc_stats_filename + '.' + save_type
desc_stats.round(3).to_csv(desc_stats_filename, sep=sep)

#get count for gender
gender_counts = emo_band_data_wide['gender'].value_counts()
with open(results_dir + '/' + 'gender_counts.txt', 'w') as f:    
    f.write('Gender    Count\n')
    for gender, count in gender_counts.items():
        f.write(f'{gender}    {count}\n')  

# %% STANDARDIZE VARIABLES FOR FOLLOWING ANALYSES
if standardize:
    scaler = StandardScaler()
    cols_to_std = ['bandwidth_negative', 'granularity_negative', 'instability_negative', 'variability_negative', 'cesd', 'rucla', 'mean_affect_level_negative', 'age']
    emo_band_data_wide[cols_to_std] = scaler.fit_transform(emo_band_data_wide[cols_to_std])

# %% REGRESSION MODELS

# 1.1: how does the emotion dynamics metrics (bandwidth, granularity, instability, variability) predict depression (cesd), when controlling for age and gender
model1 = smf.ols("cesd ~ bandwidth_negative + granularity_negative + instability_negative + variability_negative + age + gender", emo_band_data_wide)
ols_result1 = model1.fit()
print(ols_result1.summary())
#save regression results to text file
if standardize:
    dep_filename = results_dir + '/' + save_reg_filename + '_depression_no_mean_affect_standardized.txt'
else:
    dep_filename = results_dir + '/' + save_reg_filename + '_depression_no_mean_affect.txt'
with open(dep_filename, 'w') as f:
    f.write(ols_result1.summary().as_text())

# 1.2: how does the emotion dynamics metrics (bandwidth, granularity, instability, variability) predict depression, when controlling for mean negative affect level, age, and gender
model2 = smf.ols("cesd ~ bandwidth_negative + granularity_negative + instability_negative + variability_negative + mean_affect_level_negative + age + gender", emo_band_data_wide)
ols_result2 = model2.fit()
print(ols_result2.summary())
# save regression results to text file
if standardize:
    dep_filename = results_dir + '/' + save_reg_filename + '_depression_standardized.txt'
else:
    dep_filename = results_dir + '/' + save_reg_filename + '_depression.txt'
with open(dep_filename, 'w') as f:
    f.write(ols_result2.summary().as_text()) 

# Test for significant difference in R-squared between model 1.1 and model 1.2
anova_results = anova_lm(ols_result1, ols_result2)
print(anova_results)

# check VIFs for regression model 1.2
cols = ['bandwidth_negative', 'granularity_negative', 'instability_negative', 'variability_negative',
        'mean_affect_level_negative', 'instability_negative', 'variability_negative', 'age', 'gender']

X = add_constant(emo_band_data_wide[cols], has_constant="add")

vifs = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
# optional: ignore/report without intercept
vifs_no_const = vifs[vifs["variable"] != "const"]

#2.1: how does the emotion dynamics metrics (bandwidth, granularity, wmssd, sd) predict loneliness (rucla), when controlling for age and gender
model1 = smf.ols("rucla ~ bandwidth_negative + granularity_negative + instability_negative + variability_negative + cesd + age + gender", emo_band_data_wide)
ols_result1 = model1.fit()
print(ols_result1.summary())
#save regression results to text file
if standardize:
    lon_filename = results_dir + '/' + save_reg_filename + '_loneliness_no_mean_affect_standardized.txt'
else:
    lon_filename = results_dir + '/' + save_reg_filename + '_loneliness_no_mean_affect.txt'    
with open(lon_filename, 'w') as f:
    f.write(ols_result1.summary().as_text())

#2.2: how does the emotion dynamics metrics (bandwidth, granularity, wmssd, sd) predict loneliness (rucla), when controlling for mean negative affect level, cesd, age and gender
model2 = smf.ols("rucla ~ bandwidth_negative + granularity_negative + instability_negative + variability_negative + cesd + mean_affect_level_negative + age + gender", emo_band_data_wide)
ols_result2 = model2.fit()
print(ols_result2.summary())
#save regression results to text file
if standardize:
    lon_filename = results_dir + '/' + save_reg_filename + '_loneliness_standardized.txt'
else:
    lon_filename = results_dir + '/' + save_reg_filename + '_loneliness.txt'
with open(lon_filename, 'w') as f:
    f.write(ols_result2.summary().as_text())

# Test for significant difference in R-squared between model 2.1 and model 2.2
anova_results = anova_lm(ols_result1, ols_result2)
print(anova_results)


# check VIFs for regression model 2.2
cols = ['cesd', 'granularity_negative', 'bandwidth_negative', 'instability_negative', 'variability_negative',
        'mean_affect_level_negative', 'instability_negative', 'variability_negative', 'age', 'gender']

X = add_constant(emo_band_data_wide[cols], has_constant="add")

vifs = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
# optional: ignore/report without intercept
vifs_no_const = vifs[vifs["variable"] != "const"]


# %% MEDIATION ANALYSES
# mediation analysis: does depression (cesd_kw) mediate the relationship between negative emotion bandwidth and loneliness (rucla_kw), controlling for mean negative affect level
med_neg = pg.mediation_analysis(data=emo_band_data_wide, x='bandwidth_negative', m='cesd', y='rucla', covar=['mean_affect_level_negative','age','gender'], alpha=0.05, n_boot=5000)
print('Mediation Analysis for Negative Affect:')
print(med_neg)
# save mediation results to text file
if standardize:
    med_neg_filename = results_dir + '/' + save_med_filename + '_negative_affect_standardized.txt'
else:
    med_neg_filename = results_dir + '/' + save_med_filename + '_negative_affect.txt'
with open(med_neg_filename, 'w') as f:
    f.write(med_neg.to_string())



# %%
