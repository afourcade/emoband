# EmoBand

Analysis code for computing emotional bandwidth and related affective dynamics metrics from experience sampling data.

The main analysis script is [emotion_bandwidth_analysis.py](emotion_bandwidth_analysis.py). It:

- computes multivariate entropy and normalized emotional bandwidth for positive and negative affect,
- computes granularity (from ICC), instability (time-weighted MSSD), and variability (SD),
- merges questionnaire and demographic variables,
- exports processed datasets,
- runs correlations, descriptive statistics, regressions, and mediation analyses,
- saves figures and analysis tables to a results folder.

## Repository Structure

- [emotion_bandwidth_analysis.py](emotion_bandwidth_analysis.py): Main preprocessing and inferential analysis pipeline.
- [figures.py](figures.py): Script to generate conceptual/illustrative figures for MSSD, SD, ICC, and bandwidth.
- [requirements.txt](requirements.txt): Python dependencies.
- [power_sensitivity_analysis.py](power_sensitivity_analysis.py): Estimates minimum-detectable effect sizes (Cohen's f^2) for regression predictors given sample size and model size.

## Requirements

- Python 3.10+ recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

Some Plotly export operations may require `kaleido` depending on your local setup:

```bash
pip install kaleido
```

## Data Layout

The analysis script expects this directory layout under `exp_dir`:

```text
<exp_dir>/
	data/
		esm_clean_trimmed.csv
		demographic_data.csv
	results/
```

By default in [emotion_bandwidth_analysis.py](emotion_bandwidth_analysis.py), `exp_dir` is hardcoded to:

```python
exp_dir = "D:/EmoBand/"
```

Update this path before running so it matches your local environment.

## Required Columns

### `esm_clean_trimmed.csv`

Required columns used by the script:

- `participant`
- `timestamp_response`
- Positive affect items:
	- `pa_joyful_rand`
	- `pa_cheerful_rand`
	- `pa_happy_rand`
	- `pa_content_rand`
	- `pa_relaxed_rand`
	- `pa_energetic_rand`
- Negative affect items:
	- `na_tense_rand`
	- `na_irritable_rand`
	- `na_worried_rand`
	- `na_low_rand`
	- `na_lonely_rand`
	- `na_abandoned_rand`
- Questionnaire variables:
	- `RUCLA_kw` (loneliness)
	- `CESD_kw` (depressive symptoms)

Notes:

- Affect items are expected on a 1-7 scale.
- `timestamp_response` should be parseable as datetime.

### `demographic_data.csv`

Required merge key and commonly used fields:

- `participant` (merge key)
- `gender` (coded as `F`/`M` in source data; converted to `0/1`)
- `age`

## What the Script Computes

For each participant and affective dimension (`positive`, `negative`), the script computes:

- `multivariate_entropy`: Shannon entropy on discretized multivariate affect states.
- `bandwidth`: normalized state-space coverage:

$$
bandwidth = \frac{2^H}{\min(n_{levels}^{k},\ n_{prompts})}
$$

where $H$ is multivariate entropy, $k$ is number of items in the dimension, and $n_{\text{levels}}=7$.

- `icc`: intraclass correlation (`ICC3k`, pingouin).
- `granularity = 1 - icc`.
- `instability`: time-interval-weighted MSSD (within-participant), summarized by valence.
- `variability`: SD, summarized by valence.
- `mean_affect_level` by valence.

Participants with granularity greater than 1 in either valence are excluded.

## Outputs

With defaults in the script (`save_type = "tsv"`), outputs are written to `<exp_dir>/results/`.

Main exports include:

- `emo_band_proc_data.tsv`: long-format processed data.
- `emo_band_proc_data_wide.tsv`: participant-level wide dataset used for analyses.
- `emo_band_correlation.png`: lower-triangle correlation heatmap with FDR-adjusted significance masking.
- `emo_band_correlation_matrix.tsv`: correlation matrix table.
- `emo_band_descriptive_stats.tsv`: descriptive statistics table.
- `emo_band_regression_results_depression_no_mean_affect_standardized.txt`
- `emo_band_regression_results_depression_standardized.txt`
- `emo_band_regression_results_loneliness_no_mean_affect_standardized.txt`
- `emo_band_regression_results_loneliness_standardized.txt`
- `emo_band_mediation_results_negative_affect_standardized.txt`
- `gender_counts.txt`

If `standardize = False`, corresponding non-standardized filenames are produced.

## How To Run

1. Install dependencies.
2. Edit `exp_dir` in [emotion_bandwidth_analysis.py](emotion_bandwidth_analysis.py).
3. Ensure expected CSV files and columns exist.
4. Run:

```bash
python emotion_bandwidth_analysis.py
```

## Optional: Figure Generation

Run [figures.py](figures.py) to generate standalone conceptual figures for:

- MSSD and SD behavior,
- ICC/granularity illustration,
- 3D state-space bandwidth visualization.

Before running, update any hardcoded output paths in that script.

## Power Sensitivity Analysis

Run [power_sensitivity_analysis.py](power_sensitivity_analysis.py) to estimate the minimum-detectable Cohen's $f^2$ for a single predictor in multiple regression, for a fixed pooled sample size (`N`) and specified number of predictors (`k`). The script prints results to stdout for common power levels (0.80, 0.90, 0.95).

Usage:

```bash
python power_sensitivity_analysis.py
```

Notes:

- The script uses `scipy` for noncentral F computations; ensure it's installed via `requirements.txt`.
- `N` is currently hardcoded in the script; edit the `N` variable to match your sample size if needed.

## Method Notes

- Correlations are corrected for multiple testing using FDR (Benjamini-Hochberg).
- Regression models are estimated with OLS (`statsmodels`).
- Mediation uses bootstrap inference (`pingouin.mediation_analysis`, `n_boot=5000`).
- Standardization is controlled via `standardize = True` in the main script.
