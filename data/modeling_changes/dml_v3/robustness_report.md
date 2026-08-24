# V3 DML Robustness and Validation Report

> These diagnostics strengthen the first-pass DML analysis but do not remove the need for a credible causal identification strategy. All results remain observational and assumption-dependent.

## Robust uncertainty

For the primary Sentinel-2 1,000 m NDVI treatment, the point estimate is -21.180373 µg/m³ per raw NDVI unit. The original influence-function standard error is 5.848786; the station-clustered standard error is 16.514335, with 95% interval [-53.548470, 11.187723]. A 2000-replicate wild cluster bootstrap gives interval [-52.465976, 10.011109].

The clustered and wild-bootstrap procedures account for within-station dependence more directly than an observation-independent standard error. With only 35 station clusters, these intervals should still be interpreted cautiously.

## Stability and falsification

The five station-held-out fold estimates range from -88.673544 to 14.407162; the fold-level standard deviation is 41.282979. Station-level slopes are exploratory because many stations have limited repeated observations; the complete table is in `station_heterogeneity.csv`.

The within-station permutation falsification distribution for the primary residualized treatment has a 2.5%–97.5% null interval of [-23.829783, -13.759952]. This is a design check under broken treatment/outcome alignment, not a test of unmeasured confounding.

## Spatial-block sensitivity

Holding out deterministic geographic station blocks defined by the median latitude and longitude gives a primary-treatment estimate of -38.253146. This is a sensitivity design with 4 blocks, not a replacement for a pre-treatment or quasi-experimental design.

## Nuisance-learner sensitivity

Replacing the primary HistGradientBoosting nuisance learners with random-forest nuisance learners gives an estimate of -23.141328 with station-clustered 95% interval [-62.833929, 16.551274]. This checks whether the headline result is driven only by one flexible learner family.

## Interpretation

The overlap table reports the empirical distribution of the residualized treatment, which is relevant to whether the treatment remains informative after adjustment. It is not a formal proof of positivity. The station and fold tables are stability diagnostics rather than independent causal estimates. The current panel can still suffer from unmeasured spatial confounding, serial dependence, treatment measurement error, and exposure/outcome simultaneity. A stronger next design would use a clearly pre-treatment exposure window or a defensible quasi-experimental source of variation, together with dependence-aware inference.

## Generated artifacts

| File | Purpose |
|---|---|
| `robustness_summary.csv` | Original, clustered, and bootstrap uncertainty for all treatments |
| `fold_stability.csv` | Station-held-out fold estimates |
| `station_heterogeneity.csv` | Exploratory station-specific residualized slopes |
| `overlap_falsification.csv` | Residualized-treatment support and within-station permutation nulls |
| `learner_sensitivity.csv` | Random-forest nuisance learner sensitivity for the primary treatment |
| `spatial_block_sensitivity.csv` | Geographic-block DML sensitivity summary |
| `spatial_block_stability.csv` | Geographic block holdout fold composition |
| `robustness_config.json` | Seeds, replicate counts, and input hashes |
| `robustness_report.md` | Human-readable methods, results, and caveats |
