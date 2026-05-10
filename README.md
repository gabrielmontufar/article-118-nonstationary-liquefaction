# Supplementary benchmark for Article 118

This folder contains the reproducible benchmark for the manuscript:

**Non-stationary liquefaction probability under groundwater and gradation changes**

## Purpose

The benchmark evaluates how time-dependent groundwater depth and gradation changes alter the probability of liquefaction in a layered soil profile. It compares:

- deterministic factor-of-safety checks;
- stationary probabilistic assessment;
- non-stationary probabilistic assessment.

## Files

- `scripts/run_118_nonstationary_liquefaction_benchmark.py`: self-contained Python script.
- `data/synthetic_layer_profile.csv`: synthetic layered profile.
- `data/liquefaction_benchmark_results.csv`: layer-time-scenario Monte Carlo results.
- `data/liquefaction_benchmark_summary.csv`: scenario-level summary.
- `data/profile_method_comparison.csv`: stationary versus non-stationary profile comparison.
- `figures/fig01_pf_time_extreme_accumulation.png`: layer probability histories.
- `figures/fig02_profile_mean_pf_by_scenario.png`: profile-average probability by groundwater scenario.
- `figures/fig03_depth_time_pf_heatmap.png`: depth-time probability map.

SVG versions of the figures are also included for editable vector use.

## Reproducibility

Run from this folder:

```powershell
C:\Users\gjm31\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe .\scripts\run_118_nonstationary_liquefaction_benchmark.py
```

Python dependencies:

- numpy
- pandas
- Pillow

The random seed is fixed as `1182026`.

## Methodological note

The profile is synthetic and transparent. It is intended to test the proposed non-stationary reliability workflow, not to claim calibration to a specific field site. Quantitative site calibration requires field or laboratory data and should be treated as future work unless a site dataset is added.
