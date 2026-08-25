# Delhi NCR PM₂.₅ — Spatial & Causal Machine Learning Research

A research pipeline for studying the relationship between **urban green cover and PM₂.₅ pollution across Delhi NCR**, with the eventual goal of identifying spatial patterns and estimating where increased green cover may contribute to particulate pollution mitigation.

> **Current Status:** Version 1 — V3 Double Machine Learning implementation, robustness validation, time-aware sensitivity, and metric audits
> **Next:** Rolling-origin temporal validation, measurement-error sensitivity, and pre-specified nonlinear dose-response analysis

---

## Research Objective

The broader research project aims to develop a **spatial causal machine learning framework** for understanding how urban green cover influences PM₂.₅ concentrations across Delhi NCR.

The planned framework combines:

* Ground-based PM₂.₅ observations from CPCB monitoring stations
* Satellite-derived environmental indicators
* Meteorological variables
* Land-use and land-cover information
* Spatial features
* Machine learning for PM₂.₅ estimation
* Causal machine learning for estimating the effect of green cover

The research will ultimately investigate questions such as:

* How does PM₂.₅ vary spatially across Delhi NCR?
* Is there a measurable relationship between vegetation and PM₂.₅?
* Does the effect of green cover vary across different parts of the city?
* Are there identifiable vegetation thresholds associated with lower PM₂.₅?
* How do meteorology, urban structure, and other confounders influence this relationship?

---

# Version 1 — V3 Double Machine Learning Analysis

The repository now contains a complete first-pass DML implementation on the frozen 2025-context V3 datasets. The work is isolated under `data/modeling_changes/dml_v3/` and does not modify the canonical datasets, train/test splits, or protected baseline results.

## Data used

| Input | Rows | Role |
|---|---:|---|
| V3 master modeling dataset | 1,615 | Reference and integrity cross-check |
| V3 training split | 1,292 | Cross-fitted DML estimation |
| V3 locked test split | 323 | External diagnostic only |

Train/test station–year–month keys do not overlap, and the split preserves the master key universe. IIT Delhi remains train-only under the existing split design.

## DML specification

The outcome is monthly station-level `pm25`. The primary treatment is `sentinel2_ndvi_mean_1000m`; sensitivity treatments are `sentinel2_ndvi_mean_500m` and `modis_ndvi_mean_1000m`. The base specification uses five-fold station-grouped cross-fitting, median imputation, and `HistGradientBoostingRegressor` nuisance models.

Controls were pre-specified from temporal/spatial context, ERA5 meteorology, 2025 population and road density, and non-vegetation Dynamic World context. Green-cover proxies, Sentinel-5P NO₂/pollution proxies, and contemporaneous MODIS/LST or gradient variables were excluded to reduce treatment-proxy and plausible post-treatment adjustment.

## Results

| Treatment | Estimate | Original 95% interval | Dependence-aware interval |
|---|---:|---:|---:|
| Sentinel-2 NDVI, 1,000 m | -21.180373 | [-32.643994, -9.716752] | [-53.548470, 11.187723] |
| Sentinel-2 NDVI, 500 m | -7.046748 | [-16.163475, 2.069980] | [-37.240398, 23.146902] |
| MODIS NDVI, 1,000 m | -17.618217 | [-30.505183, -4.731252] | [-48.456363, 13.219928] |

The dependence-aware intervals use station-clustered uncertainty and are the preferred checks for the station-month panel. The primary point estimate is negative, but the robust interval includes zero; the result is suggestive observational evidence rather than a conclusive causal claim.

## Robustness and validation

The robustness extension includes a 2,000-replicate wild cluster bootstrap, fold and station stability tables, residualized-treatment overlap diagnostics, within-station permutation falsification, random-forest nuisance-learner sensitivity, and deterministic geographic-block cross-fitting sensitivity. The geographic-block sensitivity estimate for the primary treatment is -38.253146.

## Pre-treatment and time-aware sensitivity

A follow-up sensitivity constructs the exact previous calendar month of each NDVI treatment within each split, preventing cross-split lag leakage. An expanding time-aware design fits nuisance models only on years earlier than each holdout year (2023, 2024, and 2025). The scored sample contains 764 out-of-fold rows per treatment and 34 represented station clusters.

| Lagged treatment | Time-aware estimate | Station-clustered 95% interval |
|---|---:|---:|
| Sentinel-2 NDVI, 1,000 m | 26.970369 | [-1.665192, 55.605930] |
| Sentinel-2 NDVI, 500 m | 19.742400 | [-6.430400, 45.915201] |
| MODIS NDVI, 1,000 m | -76.421049 | [-111.513209, -41.328890] |

These lagged results differ from the contemporaneous Version 1 estimates, demonstrating that exposure timing and satellite product choice are first-order modeling decisions. They are sensitivity evidence and are not pooled with the headline estimate.

Static figures are generated under `data/modeling_changes/dml_v3/figures/`, including `dml_estimates_forest.png`, `time_aware_lagged_forest.png`, and `time_aware_sample_coverage.png`.

### DML diagnostic figures

![Original and dependence-aware DML intervals](data/modeling_changes/dml_v3/figures/dml_estimates_forest.png)

*Figure: Original influence-function intervals compared with station-clustered intervals.*

![Time-aware lagged-treatment estimates](data/modeling_changes/dml_v3/figures/time_aware_lagged_forest.png)

*Figure: Expanding time-aware DML using exact previous-calendar-month NDVI treatments.*

![Time-aware sample coverage](data/modeling_changes/dml_v3/figures/time_aware_sample_coverage.png)

*Figure: Holdout-row coverage for each expanding time-aware split.*

Validation scripts are provided for the base DML, robustness package, pre-treatment package, and attachment-driven audit package. Exact input SHA-256 hashes are recorded in the configuration files, and all generated outputs remain in the isolated DML directory.

## Attachment-driven audits

The attachment audit records missingness by split, variable domain, station, and treatment. It also logs cross-fitted nuisance-model RMSE, MAE, and R2, residual orthogonality correlation, mean orthogonal score, and a fixed PM2.5 concentration-band agreement metric for presentation readability. The band metric is descriptive only and is not an official AQI accuracy measure.

Forward-fill, backward-fill, linear interpolation, and inverse-distance imputation are not applied automatically because imputing observed outcomes or treatments can change the causal estimand and introduce artificial temporal or spatial signal. Threshold extraction is also deferred: the current partially linear DML estimator targets a constant marginal effect, whereas thresholds require a separate nonlinear dose-response specification and overlap analysis.

The attachment-to-repository implementation map is documented in `data/modeling_changes/dml_v3/attachment_implementation_scope.md`.

## Learner benchmark

A transparent nuisance-learner benchmark compares HistGradientBoosting, Random Forest, and Extra Trees using the same station-grouped folds. Learners are selected only by cross-fitted nuisance prediction RMSE, never by causal coefficient sign or confidence-interval width.

| Learner | Outcome RMSE | Outcome R2 | Treatment RMSE | Treatment R2 |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 25.054989 | 0.860111 | 0.083361 | 0.467057 |
| Random Forest | 24.460314 | 0.866673 | 0.086756 | 0.422758 |
| Extra Trees | 24.203200 | 0.869461 | 0.084366 | 0.454122 |

The predictive winners produce a sensitivity coefficient of -25.996936 with station-clustered 95% interval [-68.265131, 16.271259]. The interval still includes zero, so improved nuisance prediction should not be presented as proof of a more precise causal effect.

## Best-defensible implementation

The analysis contract in `analysis_contract.md` freezes the estimand, protected inputs, leakage rules, inference hierarchy, and specification-selection decisions before comparison. The preferred implementation is the pre-specified station-grouped DML with dependence-aware inference, time-aware sensitivities, overlap and falsification diagnostics, and transparent learner benchmarking. No specification is selected using a favorable coefficient sign, p-value, or confidence-interval width.

A rolling-origin sensitivity uses 36 chronological holdout months from 2023–2025. Every holdout month is predicted using only strictly earlier calendar months. It scores 969 rows and 35 station clusters. The primary treatment estimate is -65.311631 with station-clustered 95% interval [-86.982328, -43.640934]. A within-station expanding-time sensitivity learns station means from earlier years only, excludes one first-appearance IIT Delhi row with no earlier station mean, and estimates -44.397641 with interval [-66.307516, -22.487766] over 968 rows and 34 station clusters.

These temporal estimates are sensitivities, not replacements for the headline estimate and are not pooled with it. Their differences demonstrate that exposure timing and station-level structure materially affect the estimand. The complete outputs are `rolling_time_summary.csv`, `rolling_time_folds.csv`, `within_station_time_summary.csv`, and `within_station_time_folds.csv`.

## Reproduce the analysis

From the repository root:

```bash
python data/modeling_changes/dml_v3/run_dml.py
python data/modeling_changes/dml_v3/validate_dml.py
python data/modeling_changes/dml_v3/robustness_checks.py
python data/modeling_changes/dml_v3/validate_robustness.py
python data/modeling_changes/dml_v3/pre_treatment_dml.py
python data/modeling_changes/dml_v3/validate_pre_treatment.py
python data/modeling_changes/dml_v3/generate_dml_figures.py
python data/modeling_changes/dml_v3/attachment_audits.py
python data/modeling_changes/dml_v3/validate_attachment_audits.py
python data/modeling_changes/dml_v3/model_selection_benchmark.py
python data/modeling_changes/dml_v3/validate_model_selection.py
python data/modeling_changes/dml_v3/rolling_time_dml.py
python data/modeling_changes/dml_v3/within_station_time_dml.py
python data/modeling_changes/dml_v3/validate_best_implementation.py
```

## Pull request

The correct cross-fork pull request is [PR #2](https://github.com/hitakshijoshi20072911/PM2.5_ACM_Research_Work/pull/2). Its base is `hitakshijoshi20072911/PM2.5_ACM_Research_Work:main`, and its head is `RidhimaKulashriz/PM2.5_ACM_Research_Work:dml-v3-implementation`. The PR contains the complete base DML work, robustness artifacts, time-aware specifications, learner benchmark, audit trail, and validators.

## Interpretation and next steps

This Version 1 DML analysis remains subject to conditional exchangeability, overlap, treatment definition, measurement, spatial dependence, serial dependence, and exposure/outcome simultaneity assumptions. The next methodological priority is a clearly pre-treatment exposure window or defensible quasi-experimental variation, with dependence-aware inference treated as the primary result.

---

# Current Phase — Phase 1

## CPCB Data Ingestion, Audit & Exploratory Analysis

The repository is currently focused on establishing a reliable **ground-truth air-quality data foundation** using CPCB monitoring data for Delhi.

### Completed / In Progress

### 1. Data Ingestion

CPCB monitoring data for multiple Delhi monitoring stations has been collected and organized by:

* Monitoring station
* Year
* Observation frequency
* Pollutant

The current dataset includes hourly observations spanning **2022–2025**, where available.

### 2. Data Quality & Audit

The pipeline performs systematic inspection of the raw monitoring data, including:

* Missing values
* Duplicate records
* Timestamp consistency
* Station-level coverage
* Pollutant availability
* Invalid or anomalous observations
* Temporal coverage
* Data completeness

### 3. Data Standardization

Raw station-level data is being transformed into a consistent structure suitable for downstream analysis.

This includes:

* Timestamp normalization
* Station metadata integration
* Column standardization
* Cleaning and validation
* Station-wise dataset generation
* Consolidation of monitoring observations

### 4. Exploratory Data Analysis

Initial analysis focuses on understanding:

* PM₂.₅ temporal variation
* Station-level differences
* Seasonal patterns
* PM₂.₅ relationships with other pollutants
* Spatial distribution of monitoring observations
* Data availability and coverage

### 5. Research Data Foundation

The repository currently contains:

```text
data/
├── raw/
├── processed/
└── ml_ready/
```

The processed datasets provide the foundation for subsequent multimodal modelling.

---

# Next Phase — Phase 2

## Multimodal Environmental Data Integration

The next stage will extend the CPCB ground observations by integrating **satellite and environmental datasets**.

Planned data sources include:

| Data Source     | Purpose                                |
| --------------- | -------------------------------------- |
| CPCB            | Ground PM₂.₅ observations              | done
| Sentinel-2      | Vegetation and land-surface indicators | done
| Sentinel-5P     | Atmospheric NO₂                        | done
| ERA5-Land       | Meteorological variables               | done
| ESA WorldCover  | Land-use / land-cover information      | done 
| OpenStreetMap   | Urban and infrastructure features      | done

The objective is to construct a **spatially aligned multimodal dataset** for Delhi NCR.

Key tasks will include:

* Spatial alignment
* Temporal aggregation
* Feature extraction
* Satellite quality control
* Missing-value handling
* Grid-based feature construction
* Ground-station to spatial-grid mapping

---

# Planned Phase 3 — PM₂.₅ Spatial Prediction

Once the multimodal dataset is established, the next stage will focus on building a machine learning model for estimating PM₂.₅ across the study area.

The planned approach includes:

```text
Multimodal Environmental Data
            ↓
Feature Engineering
            ↓
Spatial Cross-Validation
            ↓
Machine Learning Model
            ↓
PM₂.₅ Spatial Surface
```

Candidate modelling approaches include **LightGBM and other tree-based machine learning methods**.

The objective is to generate a spatially continuous PM₂.₅ estimation framework rather than relying only on monitoring-station locations.

---

# Planned Phase 4 — Causal Analysis of Green Cover

The final research stage will move beyond correlation and prediction toward **causal effect estimation**.

The planned framework will investigate the effect of green cover while accounting for relevant environmental and urban confounders.

The intended methodology includes:

* Double Machine Learning (DML)
* Causal Forests / CausalForestDML
* Heterogeneous treatment-effect estimation
* Spatial heterogeneity analysis

Conceptually:

```text
Green Cover
     │
     ├──────────────► PM₂.₅
     │
     │
Confounding Factors
     │
     ├── Meteorology
     ├── Traffic / Urban Structure
     ├── Land Use
     └── Other Environmental Variables
```

The goal is to estimate **where and under what conditions green cover may have a stronger or weaker association with PM₂.₅ reduction**, rather than simply reporting a global correlation.

---

# Planned Phase 5 — Spatial Interpretation

The final stage will focus on interpreting the model outputs spatially.

Potential analyses include:

* Spatial heterogeneity of green-cover effects
* Vegetation thresholds
* Local treatment effects
* Urban morphology
* Traffic-related moderation
* Landscape configuration
* Identification of areas where additional green cover may have greater potential impact

Results will be presented through spatial visualizations and research-oriented dashboards.

---

# Repository Structure

```text
PM2.5_ACM_Research_Work/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── ml_ready/
│   └── modeling_changes/
│       └── dml_v3/        # Version 1 DML code, outputs, reports, and audits
│
├── notebooks/
│   └── Exploratory and research notebooks
│
├── reports/
│   └── Analysis outputs and research reports
│
├── src/
│   └── Data processing and modelling pipeline
│
├── Delhi_NCR_PM25_Research_Dashboard.html
├── delhi_ndvi_test.html
├── delhi_no2_test.html
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Research Pipeline

The overall planned workflow is:

```text
CPCB Ground Observations
          │
          ▼
Data Audit & Cleaning
          │
          ▼
Satellite + Meteorological Integration
          │
          ▼
Spatial Feature Engineering
          │
          ▼
PM₂.₅ Prediction Model
          │
          ▼
Spatial PM₂.₅ Surface
          │
          ▼
Causal ML
          │
          ▼
Green Cover Effect Estimation
          │
          ▼
Spatial Heterogeneity & Threshold Analysis
```

---

# Current Repository Status

| Component                        | Status                  |
| -------------------------------- | ----------------------- |
| CPCB data ingestion              | ✅ Implemented           |
| Data organization                | ✅ Implemented           |
| Data auditing                    | ✅ Implemented           |
| Data cleaning / standardization  | ✅ Implemented / ongoing |
| CPCB exploratory analysis        | ✅ Implemented / ongoing |
| Multimodal satellite integration | ✅ Included in V3 dataset |
| Meteorological integration       | ✅ Included in V3 controls |
| Spatial feature engineering      | ✅ Included in V3 dataset |
| PM₂.₅ ML prediction              | 🔄 Separate modeling track |
| Spatial PM₂.₅ surface            | 🔄 Planned refinement |
| Causal ML                        | ✅ V3 DML Version 1 |
| Green-cover effect estimation    | ✅ Version 1 complete |
| Spatial heterogeneity analysis   | ✅ Exploratory sensitivity |
| Threshold analysis               | 🔄 Planned |

---

## Development Status

This repository represents an **active research project**. The data engineering foundation, multimodal V3 modeling inputs, and a Version 1 DML analysis are now represented in the repository.

The V3 DML result is an initial observational analysis with explicit robustness caveats. It should not be treated as a final causal claim. Further work will improve the temporal design, dependence-aware inference, spatial heterogeneity analysis, and causal identification strategy.

Each subsequent phase will be added with corresponding code, datasets, validation results, and research outputs.

---

## Environment Setup

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

The repository contains the project's working datasets, including raw and processed CPCB observations and the V3 multimodal modeling inputs.

Large files are managed using **Git LFS**. The DML outputs are kept separately under `data/modeling_changes/dml_v3/` so generated diagnostics remain reviewable without changing the canonical source data policy.

The research dataset will continue to evolve as additional satellite, meteorological, land-cover, and spatial datasets are incorporated.

---

## Author

**Hitakshi Joshi**

Delhi NCR PM₂.₅ Spatial & Causal Machine Learning Research

---

> **Research status:** Version 1 V3 DML implemented and validated → next: pre-treatment panel and spatial causal refinement
