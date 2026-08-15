# Delhi NCR PM₂.₅ — Spatial & Causal Machine Learning Research

A research pipeline for studying the relationship between **urban green cover and PM₂.₅ pollution across Delhi NCR**, with the eventual goal of identifying spatial patterns and estimating where increased green cover may contribute to particulate pollution mitigation.

> **Current Status:** Phase 1 — CPCB Ground-Station Data Ingestion, Auditing & Exploratory Analysis
> **Next:** Phase 2 — Multimodal Satellite + Meteorological Data Integration

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
│   └── ml_ready/
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
| Multimodal satellite integration | 🔄 Next phase           |
| Meteorological integration       | 🔄 Planned              |
| Spatial feature engineering      | 🔄 Planned              |
| PM₂.₅ ML prediction              | 🔄 Planned              |
| Spatial PM₂.₅ surface            | 🔄 Planned              |
| Causal ML                        | 🔄 Planned              |
| Green-cover effect estimation    | 🔄 Planned              |
| Spatial heterogeneity analysis   | 🔄 Planned              |
| Threshold analysis               | 🔄 Planned              |

---

## Development Status

This repository represents an **active research project**. The current implementation should be considered the **data engineering and exploratory foundation** of the larger spatial causal machine learning framework.

The modelling and causal inference components described above are **planned subsequent phases and are not yet represented as completed results**.

As development progresses, each phase will be added to the repository with corresponding code, datasets, notebooks, validation results, and research outputs.

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

The repository contains the project's working datasets, including raw and processed CPCB observations.

Large files are managed using **Git LFS**.

The research dataset will continue to evolve as additional satellite, meteorological, land-cover, and spatial datasets are incorporated.

---

## Author

**Hitakshi Joshi**

Delhi NCR PM₂.₅ Spatial & Causal Machine Learning Research

---

> **Research status:** Phase 1 completed / ongoing refinement → Phase 2 multimodal environmental data integration
