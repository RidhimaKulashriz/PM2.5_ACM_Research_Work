PART 1: WORLDPOP AUDIT
That is a flawless execution. You should flag and take note of the no_negative_density metric value of 1126.456 in your validation table.

This indicates that the absolute lowest population density across all your station buffers (likely at a peripheral station like Dr._Karni_Singh_Shooting_Range or Aya_Nagar) is still ~1,126 people per km². This perfectly reflects reality: even the "greenest" or most peripheral edges of Delhi NCR maintain significant population density. Furthermore, you have exactly 1,615 rows, 35 stations, and zero missing values. We are ready to move on.

# 1. missing data centres from the master_modelling.csv : since 35 stations are recognized from the 44 extracted ones during cpcb working : 
1. What actually happened to the stations?

Your CPCB master has 44 stations, not 45, in this exact file. It has:

1,377,696 hourly records
44 unique stations
0 exact duplicate rows
0 duplicate station + Timestamp
158 station-year combinations

So the CPCB source itself is not suffering from duplication.

The nine stations missing from your ML-ready dataset fall into three very different categories.

Category A — good CPCB coverage: TRY TO RECOVER

These three are absolutely worth investigating:

Station	2022–25 PM₂.₅ coverage	My decision
CPRI_Mathura_Road	Very good overall; only major degradation Aug–Oct 2024	Try to recover
NSIT_Dwarka	Very good across all four years	Try to recover
Pusa	Very good across all four years	Try to recover

For example, CPRI has thousands of valid PM₂.₅ observations in every year, and NSIT/Pusa have strong coverage throughout 2022–2025.

These stations should not have been rejected merely because of CPCB incompleteness.

Category B — poor but not completely useless: DO NOT RECOVER NOW

Lodhi_Road

This one is substantially weaker:

2022 has several very incomplete months and some zero-PM₂.₅ months.
2023 is especially poor.
2024 is mixed.
2025 is better.

I would not spend time restoring Lodhi Road for the first modeling run.

It could be included later as a sensitivity analysis if you define an explicit station-month quality threshold.

Category C — unusable in the current study period: EXCLUDE

These four:

Commonwealth_Sports_Complex
IGNOU_Maidan_Garhi
JNU
NSUT_Jaffarpur
Talkatora_Garden

Actually that's five, and all five have only 2025 records with 0 valid PM₂.₅ values.

They provide no usable PM₂.₅ outcome for your 2022–2025 analytical period.

Do not try to recover them.

There is no statistical reason to add a station whose target variable is completely absent.

2. So should you expand from 35 to 39?

My answer is:

Yes, potentially — but only for CPRI Mathura Road, NSIT Dwarka and Pusa, and only if their complete satellite feature stack can be generated cleanly.

That is the important condition.

We should not add these three merely because CPCB is good.

We need:

CPCB PM2.5
+
Sentinel-2
+
Sentinel-5P
+
MODIS
+
same spatial buffers
+
same temporal coverage

for those stations.

If those three can be regenerated without rebuilding the entire satellite pipeline from scratch, I would bring them in.

That would give you:

38 stations, not 39.

35 existing + 3 recovered = 38.

Lodhi remains excluded.

The five zero-target stations remain excluded.

# osm_roads.py 
This is a successful OSM extraction, and the validation output is much cleaner than the earlier Overpass attempts. There are, however, two things I would flag before you freeze this dataset.

1. OSM itself: ✅ successful

You have:

1,600,339 highway geometries extracted
0 invalid/incomplete geometries skipped
58,713 major-road geometries
EPSG:32643 used for metric calculations
all 35 stations processed
0 missing feature values
0 duplicate station-month keys
final output exactly 1,615 rows
major_road_density_1000m <= road_density_1000m passes

So technically:

OSM PBF ✅
PyOsmium ✅
geometry conversion ✅
projection ✅
35 stations ✅
100/250/500/1000 m ✅
panel alignment ✅
validation ✅

Your final feature file is therefore usable as an input feature table.

2. The thing I WOULD flag: 1.6 million road geometries

This number:

1,600,339 highway geometries

is large, but not automatically wrong.

You downloaded the whole Northern Zone PBF, and your extraction currently retains essentially all relevant highway ways in that extract, even though your actual study region is only the Delhi-NCR station extent.

That means you are carrying a huge amount of road data through memory simply to calculate features around 35 stations.

It worked, so it's not a failure. But scientifically/engineering-wise, I'd make one improvement later:

spatially restrict the extracted road geometries to your study region before the station-buffer calculations.

It won't change the intended feature definitions; it just reduces irrelevant geometries.

I would not redo the extraction right now, because you already have valid output and you're trying to move toward modeling.

3. The more important research issue: your OSM snapshot

You are using:

January 2025 OSM snapshot

for a panel covering:

2022–2025

That is acceptable as a static infrastructure proxy, but it needs to be explicitly documented in your methodology.

It does not mean:

road density in 2022 = road density in 2025.

It means:

We use a January 2025 mapped road-network layer to characterize structural transportation infrastructure around each station, and treat that infrastructure as static over the study panel.

This is an important limitation, but it doesn't invalidate the feature.

I would not now download separate 2022/2023/2024/2025 OSM files. That adds substantial complexity for very little benefit at this stage.

4. The actual road-density numbers don't immediately concern me

You got:

minimum road density = 0.020935 km/km²
maximum road density = 43.110650 km/km²

and:

major road density:
min = 0
max = 7.431889 km/km²

Those ranges deserve a sanity check, but they aren't inherently impossible.

In particular, because you're measuring:

road centerline length / buffer area

the density can be quite high around dense urban road networks.

I would not reject the data merely because the maximum is 43.1.

What I would do before modeling

Open:

data/05_validation/osm_validation.csv

and also inspect:

data/03_features/feat_osm_roads.csv

sorted by:

road_density_100m
road_density_1000m

If you see one or two stations that are wildly different from their surrounding urban context, we investigate those geometries.

But don't let a numerical suspicion alone make us discard the dataset.

5. There's one subtle issue I'd check before v2

Your OSM extraction includes a very broad set of road classes.

That's reasonable for total road density, but remember what your feature means.

You currently have:

road_density_100m
road_density_250m
road_density_500m
road_density_1000m
major_road_density_1000m

I like this feature set.

I would not add more OSM variables now.

For your baseline models, these five are sufficient as the urban transportation layer

# landcover static baseline of 2021 : 
Important distinction for your paper

You should describe WorldCover like this:

"ESA WorldCover 2021 was used as a high-resolution static baseline describing surrounding land-use/land-cover composition. It was not treated as a time-varying exposure for 2022–2025."

And the limitation:

"Because the WorldCover map represents 2021 conditions, temporal land-cover changes during 2022–2025 are not directly represented."

That's honest and defensible.

It is also consistent with the availability of the product: ESA currently provides WorldCover 2020 and 2021 maps, with the 2021 v200 product at 10 m resolution.

1. Verification of ESA/WorldCover/v200

Accessibility: The product is public and accessible in Google Earth Engine via the ESA/WorldCover/v200 ImageCollection.

Temporal Coverage: v200 explicitly maps the global land cover for the year 2021.

Spatial Resolution: 10 meters, derived primarily from Sentinel-1 and Sentinel-2 data.

Class Codes: Verified. 10 (Tree), 20 (Shrub), 30 (Grass), 40 (Crop), 50 (Built), 60 (Bare), 70 (Snow), 80 (Water), 90 (Wetland), 95 (Mangroves), 100 (Moss). We will extract 50, 30, 40, and 80.

2. Why Categorical Pixel Counting Instead of Numerical Averaging?
WorldCover Map values are nominal categories, not ordinal or continuous variables. Calculating the mathematical mean of a raster containing Built-up (50) and Grassland (30) pixels yields a value of 40 (Cropland). This is mathematically meaningless and scientifically false. The only valid aggregation for land-use maps is compositional: counting the spatial frequency (pixels) of each distinct category and calculating its fractional proportion of the total valid area.

3. Why a 2021 Static Baseline is Scientifically Acceptable
For causal inference (particularly Double Machine Learning frameworks), distinguishing between pre-treatment confounders and post-treatment mediators is critical. While NDVI/EVI dynamically measure vegetative greenness (the core environmental exposure), WorldCover provides the underlying structural land-use composition (built environment vs. agricultural vs. water). Using a 2021 pre-study baseline ensures this spatial context is independent of post-2021 temporal variations, eliminating the risk of post-treatment bias or circular causality in the adjustment set.

4. Scientific Limitation
ESA WorldCover 2021 is used as a static pre-study land-cover baseline. It does not represent land-cover changes during 2022–2025. With a reported global accuracy of ~76.7%, the dataset functions as a contextual spatial baseline for the 35 monitoring stations, rather than a perfect parcel-level land-use ground truth.

5. Exact Output Files

src/features/build_worldcover.py

data/03_features/feat_worldcover.csv

data/05_validation/worldcover_validation.csv

# iit delhi problem : 
"The final panel contained 35 monitoring stations. One station (IIT Delhi) had only one valid multimodal station-month observation and was therefore retained in the analytical dataset but assigned exclusively to the training subset during the primary temporal-stratified holdout; it was not used as an independent test location."

### dataset creation and splitting info (v1) [failed logic btw]: 
Yes — this split has **passed all the hard validation checks**, so you can now treat the generated train/test files as your primary modeling split.

The important outputs and their purposes are:

| File                                                 | What it is                            | Use                                                                                                                                                    |
| ---------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/ml_ready/master_modelling_dataset.csv`         | **V1 / original ML-ready master**     | Immutable historical baseline. Do not modify.                                                                                                          |
| `data/ml_ready/master_modelling_dataset_v2.csv`      | **Final integrated modeling dataset** | Canonical dataset containing V1 + ERA5 + WorldPop + OSM + WorldCover. Use this for final dataset-level auditing and as the source for modeling splits. |
| `data/modeling/splits/train.csv`                     | **FINAL TRAINING SET**                | Use this as the training input for Linear/Ridge, Random Forest, LightGBM, etc.                                                                         |
| `data/modeling/splits/test.csv`                      | **FINAL TESTING SET**                 | Locked holdout for final unbiased evaluation after model fitting/tuning. Do not train or tune on it.                                                   |
| `data/modeling/splits/split_manifest.csv`            | Split metadata/provenance             | Records seed, row counts, train/test proportions, stations/years, and the IIT_Delhi train-only exception.                                              |
| `data/modeling/splits/year_split_diagnostics.csv`    | Year-level split audit                | Check that 2022, 2023, 2024, 2025 are represented properly in train/test.                                                                              |
| `data/modeling/splits/station_split_diagnostics.csv` | Station-level split audit             | Shows train/test counts per station and documents `IIT_Delhi` as train-only.                                                                           |
| `data/modeling/splits/season_split_diagnostics.csv`  | Seasonal split audit                  | Useful for checking winter/summer/monsoon/post-monsoon balance.                                                                                        |

### Your actual final files

**Training:**

```text
data/modeling/splits/train.csv
```

**Testing:**

```text
data/modeling/splits/test.csv
```

These are the two files you should feed into the modeling pipeline.

### What the successful output proves - it failedd

Your terminal shows:

```text
V2 rows: 1615

Train rows: 1292
Test rows: 323

Train fraction: 0.800000
Test fraction: 0.200000

Train stations: 35
Test stations: 34

Train years: 2022, 2023, 2024, 2025
Test years: 2022, 2023, 2024, 2025

Train-only stations: IIT_Delhi
Key overlap: 0
Row universe preserved: TRUE

FINAL STATUS:
PASS
```

So the primary split satisfies the design we settled on:

```text
1,615 total
      ↓
1,292 train (80%)
  323 test  (20%)
```

and importantly, **every year appears in both sets**.

`IIT_Delhi` is train-only because V2 contains only one valid station-month observation for it. That is an explicitly documented data-coverage limitation, not a splitting error.

### What you should not do now

Do **not** regenerate V2 from the train/test files, and do not merge train and test back into a new modeling dataset.

The hierarchy should remain:

```text
master_modelling_dataset_v2.csv   ← canonical final dataset
                ↓
        prepare_and_split.py
          ↙             ↘
     train.csv       test.csv
```

Then modeling happens **only on `train.csv` initially**. The `test.csv` stays untouched until the final model comparison/evaluation.

One other note: the `PerformanceWarning: DataFrame is highly fragmented` is a performance warning, not a correctness failure. Your split itself passed. You do not need to redo the split because of that warning.

# splitting decision and logic (v2) : 
Yes. The logic behind the script is basically: make the 20% test set representative across time and stations while preserving the exact row universe and the exact 80:20 ratio.

But I would still ask Gemini Pro to critically review it before you lock the split, because there is a genuine tradeoff between year/month representativeness, station representation, and exact 80:20 allocation. The script I gave you is a defensible approach, but it is not the only possible approach, and for a research paper I want the rationale to be explicit.

1. Logic behind the current splitting strategy

Your data are not ordinary i.i.d. tabular data. They are a station-month panel:

station × year × month

with:

1615 observations
35 stations
2022–2025

and the target:

pm25

The original concern from your teacher was that a chronological split such as:

2022–2023 → train
2024–2025 → test

could be misleading because different years may represent different environmental regimes. Your example of the COVID/lockdown year illustrates exactly that concern.

So we wanted the test set to contain observations from all four years, rather than making “year” itself equivalent to “train vs test.”

The current script therefore does this:

First: it treats

data/ml_ready/master_modelling_dataset_v2.csv

as immutable.

Second: it validates the V2 dataset before splitting:

1,615 rows
35 stations
2022–2025
unique station-year-month keys
numeric/non-missing PM₂.₅
stable station coordinates
no infinite values

Third: it calculates the overall exact test size:

1615 × 0.20 = 323

so:

Train = 1292
Test  = 323

Fourth: it allocates the test quota across year × month cells, rather than simply randomly taking 20% of the whole dataset.

That matters because your earlier split produced:

Winter       56
Summer        2
Monsoon       4
Post-monsoon 261

which was obviously a terrible test distribution.

The current script instead tries to make the test set cover the calendar structure.

Fifth: it guarantees one test observation from every station with at least two observations.

That was introduced because you discovered:

IIT_Delhi = 1 observation

and therefore it is mathematically impossible for IIT_Delhi to occur in both train and test without duplicating its only observation.

So:

IIT_Delhi → train only

while the other stations with ≥2 observations get test representation.

Finally: it validates the resulting split and only writes the files if everything passes.

2. Why this is better than the previous split

Your previous split technically gave:

Train = 1344
Test = 271

instead of 80:20, and the test set was heavily concentrated in post-monsoon observations.

The resulting LightGBM performance was:

Test R²  = -0.421
RMSE      = 79.50
MAE       = 53.76

while training R² was:

0.999

and year-wise performance collapsed in 2023/2024.

That was a strong sign that the evaluation distribution needed to be investigated before interpreting the models.

3. But there is an important research question about the current strategy

The current method prioritizes:

1. exact 80:20
2. year-month representation
3. station representation

That is reasonable.

However, forcing a test observation from every station and every non-degenerate year-month cell is a constraint, and constraints can sometimes make the test set less proportional than a pure stratified sample.

For example, suppose a particular year-month already has many observations. The algorithm may still be forced to allocate at least one test row there and then redistribute the remaining quota elsewhere.

So I would like Gemini to evaluate whether a more statistically principled approach would be:

proportional stratified sampling by year-month first, followed by minimal station-coverage swaps

rather than building the split around station guarantees first.

That is why I would get a second opinion before freezing it.

4. The deeper issue: there are actually TWO validation questions

This is very important for your paper.

Your current split is intended to answer:

How well does the model predict PM₂.₅ when all study years and seasons are represented in both training and testing?

That is a valid primary predictive experiment.

But it does not answer:

Can the model generalize to a completely new monitoring station?

because the same stations mostly exist in both sets.

For that, we need a separate station-grouped/spatial validation later.

So I would eventually have:

Experiment A — primary balanced holdout
80:20
year/month stratified
same stations may appear in both
Experiment B — spatial generalization
Group by station
some stations entirely held out

The second one is especially important for your geospatial research.

### something new  change of dataset for population and land cover: 
(venv) PS C:\Users\Hitakkshi Joshi\Desktop\acm slot 11> python src\features\phase0_temporal_update.py
======================================================================
PHASE 0 — CLEAN 2025 CONTEXTUAL UPDATE
======================================================================
Loaded V2: 1615 rows, 172 columns.
Initialising Google Earth Engine...
Google Earth Engine initialized successfully.

============================================================
2025 LAND COVER — GOOGLE DYNAMIC WORLD
============================================================
Dynamic World 2025 images: 2232477
Processing Dynamic World: Alipur (1/35)
Processing Dynamic World: Anand_Vihar (2/35)
Processing Dynamic World: Ashok_Vihar (3/35)
Processing Dynamic World: Aya_Nagar (4/35)
Processing Dynamic World: Bawana (5/35)
Processing Dynamic World: Burari_Crossing (6/35)
Processing Dynamic World: Chandni_Chowk (7/35)
Processing Dynamic World: DTU (8/35)
Processing Dynamic World: Dr._Karni_Singh_Shooting_Range (9/35)
Processing Dynamic World: Dwarka-Sector_8 (10/35)
Processing Dynamic World: IGI_Airport_(T3) (11/35)
Processing Dynamic World: IHBAS_Dilshad_Garden (12/35)
Processing Dynamic World: IIT_Delhi (13/35)
Processing Dynamic World: IMD_Lodhi_Road (14/35)
Processing Dynamic World: ITO (15/35)
Processing Dynamic World: Jahangirpuri (16/35)
Processing Dynamic World: Jawaharlal_Nehru_Stadium (17/35)
Processing Dynamic World: Major_Dhyan_Chand_National_Stadium (18/35)
Processing Dynamic World: Mandir_Marg (19/35)
Processing Dynamic World: Mundka (20/35)
Processing Dynamic World: Najafgarh (21/35)
Processing Dynamic World: Narela (22/35)
Processing Dynamic World: Nehru_Nagar (23/35)
Processing Dynamic World: North_Campus_DU (24/35)
Processing Dynamic World: Okhla_Phase-2 (25/35)
Processing Dynamic World: Patparganj (26/35)
Processing Dynamic World: Punjabi_Bagh (27/35)
Processing Dynamic World: R_K_Puram (28/35)
Processing Dynamic World: Rohini (29/35)
Processing Dynamic World: Shadipur (30/35)
Processing Dynamic World: Sirifort (31/35)
Processing Dynamic World: Sonia_Vihar (32/35)
Processing Dynamic World: Sri_Aurobindo_Marg (33/35)
Processing Dynamic World: Vivek_Vihar (34/35)
Processing Dynamic World: Wazirpur (35/35)

============================================================
2025 POPULATION — JRC GHSL P2023A
============================================================
Processing GHSL population: Alipur (1/35)
Processing GHSL population: Anand_Vihar (2/35)
Processing GHSL population: Ashok_Vihar (3/35)
Processing GHSL population: Aya_Nagar (4/35)
Processing GHSL population: Bawana (5/35)
Processing GHSL population: Burari_Crossing (6/35)
Processing GHSL population: Chandni_Chowk (7/35)
Processing GHSL population: DTU (8/35)
Processing GHSL population: Dr._Karni_Singh_Shooting_Range (9/35)
Processing GHSL population: Dwarka-Sector_8 (10/35)
Processing GHSL population: IGI_Airport_(T3) (11/35)
Processing GHSL population: IHBAS_Dilshad_Garden (12/35)
Processing GHSL population: IIT_Delhi (13/35)
Processing GHSL population: IMD_Lodhi_Road (14/35)
Processing GHSL population: ITO (15/35)
Processing GHSL population: Jahangirpuri (16/35)
Processing GHSL population: Jawaharlal_Nehru_Stadium (17/35)
Processing GHSL population: Major_Dhyan_Chand_National_Stadium (18/35)
Processing GHSL population: Mandir_Marg (19/35)
Processing GHSL population: Mundka (20/35)
Processing GHSL population: Najafgarh (21/35)
Processing GHSL population: Narela (22/35)
Processing GHSL population: Nehru_Nagar (23/35)
Processing GHSL population: North_Campus_DU (24/35)
Processing GHSL population: Okhla_Phase-2 (25/35)
Processing GHSL population: Patparganj (26/35)
Processing GHSL population: Punjabi_Bagh (27/35)
Processing GHSL population: R_K_Puram (28/35)
Processing GHSL population: Rohini (29/35)
Processing GHSL population: Shadipur (30/35)
Processing GHSL population: Sirifort (31/35)
Processing GHSL population: Sonia_Vihar (32/35)
Processing GHSL population: Sri_Aurobindo_Marg (33/35)
Processing GHSL population: Vivek_Vihar (34/35)
Processing GHSL population: Wazirpur (35/35)

============================================================
2025 OSM ROAD INFRASTRUCTURE
============================================================

============================================================
ASSEMBLING MASTER MODELING DATASET V3
============================================================
Removing old contextual columns:
  - worldpop_density_250m
  - worldpop_density_500m
  - worldpop_density_1000m
  - road_density_100m
  - road_density_250m
  - road_density_500m
  - road_density_1000m
  - major_road_density_1000m
  - worldcover_2021_built_frac_100m
  - worldcover_2021_grass_frac_100m
  - worldcover_2021_cropland_frac_100m
  - worldcover_2021_water_frac_100m
  - worldcover_2021_built_frac_250m
  - worldcover_2021_grass_frac_250m
  - worldcover_2021_cropland_frac_250m
  - worldcover_2021_water_frac_250m
  - worldcover_2021_built_frac_500m
  - worldcover_2021_grass_frac_500m
  - worldcover_2021_cropland_frac_500m
  - worldcover_2021_water_frac_500m
  - worldcover_2021_built_frac_1000m
  - worldcover_2021_grass_frac_1000m
  - worldcover_2021_cropland_frac_1000m
  - worldcover_2021_water_frac_1000m

V3 saved to:
C:\Users\Hitakkshi Joshi\Desktop\acm slot 11\data\modeling_changes\datasets\master_modeling_dataset_v3.csv

============================================================
GENERATING YEAR × MONTH STRATIFIED SPLIT
============================================================

======================================================================
PHASE 0 COMPLETE — ALL DATA ARE ACTUAL SOURCED VALUES
======================================================================
V2 rows: 1615
V3 rows: 1615
Train rows: 1292
Test rows: 323

2025 contextual layers:
✓ Dynamic World 2025 — genuine GEE extraction
✓ GHSL 2025 population — genuine GEE extraction
✓ OSM 2025 — existing validated project feature table

2021 contextual layers removed:
✓ ESA WorldCover 2021 removed
✓ WorldPop 2021-era features removed

Canonical datasets modified:
NO

Output:
C:\Users\Hitakkshi Joshi\Desktop\acm slot 11\data\modeling_changes\datasets\master_modeling_dataset_v3.csv
C:\Users\Hitakkshi Joshi\Desktop\acm slot 11\data\modeling_changes\splits\train.csv
C:\Users\Hitakkshi Joshi\Desktop\acm slot 11\data\modeling_changes\splits\test.csv

STATUS: PASS
======================================================================
(venv) PS C:\Users\Hitakkshi Joshi\Desktop\acm slot 11> 

## resplitting of data (v3): 
Yes. **This split is finally good enough to freeze and use.** I would stop changing the split now and move to baseline modelling.

### Why I’m comfortable with it

**Station balance — good**

* Every normal station has **9 or 10 test observations**.
* 47-row stations have 9, 45-row station has 9, Chandni Chowk has 7.
* IIT Delhi is correctly train-only.
* No station is dominating the test set.

**Year balance — good**

* 2022: 20.05%
* 2023: 19.65%
* 2024: 19.70%
* 2025: 20.59%

That is very close to the intended 20%.

**Month balance — good**
The test fractions range roughly from **18.8% to 20.7%**, which is a very reasonable result given integer row constraints.

**Year × Month balance — especially good**
Every available year-month combination is represented, and the test fraction is essentially:

* ~18.2% for 33-row strata
* ~20.0–20.6% for 34–35-row strata
* no extreme over/under-representation

This is much better than your earlier splits.

### One small technical correction

Your `expected_test` is calculated using:

```text
323 / 1615 = 19.999%
```

but your actual eligible sampling fraction is:

```text
323 / 1614 = 20.0124%
```

because IIT Delhi is deliberately excluded from test.

This is **not a problem with the split**. It is only a diagnostic-definition issue. For the final report, describe the split as:

> **1,292 training observations and 323 test observations (approximately 80:20), with IIT Delhi retained exclusively in training because only one valid station-month observation was available.**

And ideally change the diagnostics' expected fraction to use the **eligible pool (1,614)** so the reported expected values are internally consistent.

### One important modelling precaution

Your `train.csv` and `test.csv` now contain the derived `season` column. For your actual models, **do not accidentally use `season` as an additional predictor if your notebook was not designed for it**. It is derived directly from `month` and is primarily useful for diagnostics/stratification. Keep the original feature-selection logic consistent with your previous baseline runs.

### Final decision

**Freeze this split. ✅**

Use it for the new V3 baseline modelling, and record this methodology:

> A station-proportional 80:20 holdout was constructed with fixed station-level test quotas. Within each station, observations were selected to preserve available temporal coverage, while candidate splits were evaluated using Year × Month, month, year and seasonal distributions. PM₂.₅ was not used during split construction. IIT Delhi was retained in training only because it contained a single valid station-month observation.

Then move on to your **baseline models → spatial validation → DML/cross-fitting**. Do not spend more time optimizing this split unless the final `validation_report.csv` contains a FAIL.
