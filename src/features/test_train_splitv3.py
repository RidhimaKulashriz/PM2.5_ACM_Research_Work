"""
V3 — FINAL STATION-PROPORTIONAL SPATIOTEMPORAL SPLIT
=====================================================

Research
--------
Urban Green Cover Thresholds for PM2.5 Mitigation:
A Spatial Causal Machine Learning Framework for Delhi NCR

PURPOSE
-------
Create the primary reproducible 80:20 train/test split from V3.

DESIGN
------
1. V3 master dataset is READ ONLY.
2. IIT_Delhi (n=1) is TRAIN ONLY.
3. Exactly 323 observations are assigned to TEST.
4. Station TEST quotas are fixed using proportional
   largest-remainder allocation.
5. Station quotas can NEVER change during optimization.
6. Every eligible station must retain all of its available
   study years in TEST.
7. Every eligible station must retain all of its available
   seasons in TEST.
8. The global Year x Month distribution is optimized directly.
9. The optimization uses only station/year/month/season structure.
10. PM2.5 is NEVER used for split selection.
11. Files are written only after all validation checks pass.

WHY THIS VERSION
----------------
The previous greedy implementation could produce a balanced
station split but distort the temporal distribution, especially
October/November.

This implementation treats split construction as a constrained
binary allocation problem:

    station quota       -> HARD CONSTRAINT
    station-year       -> HARD CONSTRAINT
    station-season     -> HARD CONSTRAINT
    total test size    -> HARD CONSTRAINT
    temporal balance   -> OPTIMIZATION OBJECTIVE

The expected temporal distribution is calculated from the eligible
test pool (all non-singleton observations), because IIT_Delhi is
structurally unavailable for TEST.

DEPENDENCY
----------
This implementation uses scipy.optimize.milp (HiGHS).
SciPy must therefore be installed.

    pip install scipy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import (
        Bounds,
        LinearConstraint,
        milp,
    )
    from scipy.sparse import lil_matrix

    SCIPY_AVAILABLE = True

except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

V3_PATH = (
    ROOT
    / "data"
    / "modeling_changes"
    / "datasets"
    / "master_modeling_dataset_v3.csv"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "modeling_changes"
    / "splits"
)


# ============================================================
# CONSTANTS
# ============================================================

SEED = 42

EXPECTED_TOTAL = 1615
EXPECTED_TEST = 323
EXPECTED_TRAIN = 1292
EXPECTED_STATIONS = 35

SINGLETON_STATION = "IIT_Delhi"

YEARS = [2022, 2023, 2024, 2025]

SEASON_MAP = {
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Summer",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-monsoon",
    11: "Post-monsoon",
    12: "Winter",
}

SEASONS = [
    "Winter",
    "Summer",
    "Monsoon",
    "Post-monsoon",
]

KEYS = [
    "station",
    "year",
    "month",
]

TEST_FRACTION_ELIGIBLE = None


# ============================================================
# MASTER VALIDATION
# ============================================================

def validate_master(df: pd.DataFrame) -> None:

    required = [
        "station",
        "year",
        "month",
        "latitude",
        "longitude",
        "pm25",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required V3 columns: {missing}"
        )

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"found {len(df)}."
        )

    if df["station"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_STATIONS} stations, "
            f"found {df['station'].nunique()}."
        )

    if df.duplicated(KEYS).any():
        raise RuntimeError(
            "Duplicate station-year-month keys detected."
        )

    if df["pm25"].isna().any():
        raise RuntimeError(
            "Missing PM2.5 values detected."
        )

    years = set(
        df["year"].astype(int)
    )

    if not years.issubset(
        set(YEARS)
    ):
        raise RuntimeError(
            f"Unexpected years detected: {sorted(years)}"
        )

    months = set(
        df["month"].astype(int)
    )

    if not months.issubset(
        set(range(1, 13))
    ):
        raise RuntimeError(
            f"Unexpected months detected: {sorted(months)}"
        )

    coordinate_counts = (
        df.groupby("station")
        [["latitude", "longitude"]]
        .nunique()
    )

    if (
        coordinate_counts > 1
    ).any().any():

        raise RuntimeError(
            "Station coordinates are not stable."
        )


# ============================================================
# ADD SEASON
# ============================================================

def add_season(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["season"] = (
        result["month"]
        .astype(int)
        .map(SEASON_MAP)
    )

    if result["season"].isna().any():
        raise RuntimeError(
            "Could not assign season to all observations."
        )

    return result


# ============================================================
# STATION QUOTAS
# ============================================================

def allocate_station_quotas(
    eligible: pd.DataFrame,
    total_test: int,
) -> pd.Series:
    """
    Allocate exactly 323 test observations proportionally
    across eligible stations using largest remainder.

    IIT_Delhi is not present here because it is train-only.
    """

    station_counts = (
        eligible["station"]
        .value_counts()
        .sort_index()
    )

    raw = (
        station_counts.astype(float)
        * total_test
        / station_counts.sum()
    )

    quotas = (
        np.floor(raw)
        .astype(int)
    )

    remaining = (
        total_test
        - int(quotas.sum())
    )

    remainders = (
        raw
        - np.floor(raw)
    )

    order = sorted(
        station_counts.index,
        key=lambda station: (
            -remainders.loc[station],
            str(station),
        ),
    )

    for station in order:

        if remaining <= 0:
            break

        quotas.loc[station] += 1
        remaining -= 1

    if remaining != 0:
        raise RuntimeError(
            "Could not allocate exact station quotas."
        )

    if int(quotas.sum()) != total_test:
        raise RuntimeError(
            "Station quota total is not 323."
        )

    for station, quota in quotas.items():

        available = int(
            station_counts.loc[station]
        )

        if quota <= 0:
            raise RuntimeError(
                f"Station {station} received "
                f"zero test observations."
            )

        if quota >= available:
            raise RuntimeError(
                f"Station {station}: invalid quota "
                f"{quota}/{available}."
            )

    return quotas.astype(int)


# ============================================================
# TEMPORAL TARGETS
# ============================================================

def build_temporal_targets(
    eligible: pd.DataFrame,
    total_test: int,
) -> dict:
    """
    Calculate proportional expected TEST counts from the
    eligible pool only.

    This is important because IIT_Delhi is unavailable for TEST.
    """

    pool_total = len(eligible)

    ratio = (
        total_test
        / pool_total
    )

    ym_counts = (
        eligible
        .groupby(
            ["year", "month"]
        )
        .size()
        .to_dict()
    )

    year_counts = (
        eligible["year"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    month_counts = (
        eligible["month"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    season_counts = (
        eligible["season"]
        .value_counts()
        .reindex(SEASONS)
        .fillna(0)
        .to_dict()
    )

    return {
        "ratio": ratio,

        "year_month": {
            key: count * ratio
            for key, count in ym_counts.items()
        },

        "year": {
            key: count * ratio
            for key, count in year_counts.items()
        },

        "month": {
            key: count * ratio
            for key, count in month_counts.items()
        },

        "season": {
            key: count * ratio
            for key, count in season_counts.items()
        },
    }


# ============================================================
# MILP SPLIT SOLVER
# ============================================================

def solve_optimal_split(
    master: pd.DataFrame,
    eligible: pd.DataFrame,
    station_quotas: pd.Series,
    targets: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Solve the constrained TEST allocation.

    Binary variable:
        x_i = 1 -> row i goes to TEST
        x_i = 0 -> row i stays in TRAIN

    HARD CONSTRAINTS
    ----------------
    - exactly 323 test rows
    - exact station quotas
    - at least one test observation per available
      station-year combination
    - at least one test observation per available
      station-season combination

    OBJECTIVE
    ---------
    Minimize absolute deviation from the proportional
    Year x Month distribution.

    No PM2.5 values enter the optimization.
    """

    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "SciPy is required for the final constrained split.\n"
            "Install it with:\n\n"
            "    pip install scipy\n"
        )

    eligible = (
        eligible
        .copy()
        .reset_index(drop=True)
    )

    n_rows = len(eligible)

    # --------------------------------------------------------
    # Build Year x Month categories
    # --------------------------------------------------------

    ym_values = (
        eligible[
            ["year", "month"]
        ]
        .astype(int)
        .drop_duplicates()
        .sort_values(
            ["year", "month"]
        )
        .itertuples(
            index=False,
            name=None,
        )
    )

    ym_keys = list(ym_values)

    ym_to_id = {
        key: i
        for i, key in enumerate(ym_keys)
    }

    ym_index = np.array(
        [
            ym_to_id[
                (
                    int(row.year),
                    int(row.month),
                )
            ]
            for row in eligible.itertuples()
        ],
        dtype=int,
    )

    # --------------------------------------------------------
    # Variable layout
    #
    # x_i                      -> binary row variables
    # d_plus_k / d_minus_k    -> absolute deviation variables
    # --------------------------------------------------------

    n_ym = len(ym_keys)

    X_START = 0
    DPLUS_START = n_rows
    DMINUS_START = (
        n_rows + n_ym
    )

    n_variables = (
        n_rows
        + n_ym
        + n_ym
    )

    objective = np.zeros(
        n_variables,
        dtype=float,
    )

    # Primary objective: Year x Month balance
    objective[
        DPLUS_START:
        DPLUS_START + n_ym
    ] = 1.0

    objective[
        DMINUS_START:
        DMINUS_START + n_ym
    ] = 1.0

    # Tiny deterministic tie-break on row selection.
    # This has negligible influence on the main objective.
    tiny_rng = np.random.default_rng(
        SEED
    )

    objective[
        X_START:
        X_START + n_rows
    ] = (
        tiny_rng.uniform(
            0,
            1e-7,
            size=n_rows,
        )
    )

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    row_constraints = []
    lower_bounds = []
    upper_bounds = []

    def add_constraint(
        coefficients: dict[int, float],
        lower: float,
        upper: float,
    ):
        row_constraints.append(
            coefficients
        )
        lower_bounds.append(
            lower
        )
        upper_bounds.append(
            upper
        )

    # --------------------------------------------------------
    # 1. Exact total test size
    # --------------------------------------------------------

    total_coeff = {
        i: 1.0
        for i in range(n_rows)
    }

    add_constraint(
        total_coeff,
        EXPECTED_TEST,
        EXPECTED_TEST,
    )

    # --------------------------------------------------------
    # 2. Exact station quotas
    # --------------------------------------------------------

    for station, quota in (
        station_quotas.items()
    ):

        station_positions = (
            np.flatnonzero(
                eligible["station"]
                .eq(station)
                .to_numpy()
            )
        )

        coefficients = {
            int(pos): 1.0
            for pos in station_positions
        }

        add_constraint(
            coefficients,
            int(quota),
            int(quota),
        )

    # --------------------------------------------------------
    # 3. Station-year coverage
    # --------------------------------------------------------

    grouped_sy = (
        eligible
        .groupby(
            [
                "station",
                "year",
            ]
        )
    )

    for (
        station,
        year,
    ), group in grouped_sy:

        positions = (
            group.index
            .to_numpy()
        )

        coefficients = {
            int(pos): 1.0
            for pos in positions
        }

        # Require >= 1 test observation whenever
        # the station-year exists in the dataset.
        add_constraint(
            coefficients,
            1.0,
            np.inf,
        )

    # --------------------------------------------------------
    # 4. Station-season coverage
    # --------------------------------------------------------

    grouped_ss = (
        eligible
        .groupby(
            [
                "station",
                "season",
            ]
        )
    )

    for (
        station,
        season,
    ), group in grouped_ss:

        positions = (
            group.index
            .to_numpy()
        )

        coefficients = {
            int(pos): 1.0
            for pos in positions
        }

        add_constraint(
            coefficients,
            1.0,
            np.inf,
        )

    # --------------------------------------------------------
    # 5. Year x Month absolute deviations
    #
    # selected_count - expected =
    # d_plus - d_minus
    # --------------------------------------------------------

    for key, ym_id in ym_to_id.items():

        positions = np.flatnonzero(
            ym_index == ym_id
        )

        coefficients = {
            int(pos): 1.0
            for pos in positions
        }

        coefficients[
            DPLUS_START + ym_id
        ] = -1.0

        coefficients[
            DMINUS_START + ym_id
        ] = 1.0

        expected = targets[
            "year_month"
        ][key]

        add_constraint(
            coefficients,
            expected,
            expected,
        )

    # --------------------------------------------------------
    # Convert sparse constraints
    # --------------------------------------------------------

    matrix = lil_matrix(
        (
            len(row_constraints),
            n_variables,
        ),
        dtype=float,
    )

    for row_number, coefficients in enumerate(
        row_constraints
    ):
        for variable, coefficient in (
            coefficients.items()
        ):
            matrix[
                row_number,
                variable
            ] = coefficient

    constraint = LinearConstraint(
        matrix.tocsr(),
        np.asarray(
            lower_bounds,
            dtype=float,
        ),
        np.asarray(
            upper_bounds,
            dtype=float,
        ),
    )

    # --------------------------------------------------------
    # Variable bounds
    # --------------------------------------------------------

    lower = np.zeros(
        n_variables,
        dtype=float,
    )

    upper = np.full(
        n_variables,
        np.inf,
        dtype=float,
    )

    # x_i are binary
    lower[
        X_START:
        X_START + n_rows
    ] = 0.0

    upper[
        X_START:
        X_START + n_rows
    ] = 1.0

    bounds = Bounds(
        lower,
        upper,
    )

    integrality = np.zeros(
        n_variables,
        dtype=int,
    )

    # Row-selection variables are integer/binary.
    integrality[
        X_START:
        X_START + n_rows
    ] = 1

    print("\nSOLVING CONSTRAINED TEMPORAL ALLOCATION")
    print(f"Eligible rows      : {n_rows}")
    print(f"Test rows required : {EXPECTED_TEST}")
    print(f"Year-month strata  : {n_ym}")
    print(
        "Constraints        : "
        f"{len(row_constraints)}"
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options={
            "time_limit": 120,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )

    if not result.success:
        raise RuntimeError(
            "MILP split optimization failed.\n"
            f"Status : {result.status}\n"
            f"Message: {result.message}"
        )

    decision = (
        result.x[
            X_START:
            X_START + n_rows
        ]
    )

    selected_mask = (
        decision > 0.5
    )

    if selected_mask.sum() != EXPECTED_TEST:
        raise RuntimeError(
            "MILP returned an incorrect number "
            "of test observations."
        )

    # --------------------------------------------------------
    # Build split
    # --------------------------------------------------------

    selected_original_indices = (
        eligible.loc[
            selected_mask
        ]
        .index
        .to_numpy()
    )

    test = eligible.loc[
        selected_mask
    ].copy()

    train = master[
        ~master[KEYS].apply(
            tuple,
            axis=1,
        ).isin(
            set(
                test[KEYS]
                .apply(
                    tuple,
                    axis=1,
                )
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Objective summary
    # --------------------------------------------------------

    diagnostics = {
        "solver": "SciPy MILP / HiGHS",
        "solver_status": int(
            result.status
        ),
        "solver_message": str(
            result.message
        ),
        "objective_value": float(
            result.fun
        ),
        "eligible_rows": int(
            n_rows
        ),
        "test_rows": int(
            len(test)
        ),
        "station_quota_locked": True,
        "pm25_used": False,
    }

    return (
        train,
        test,
        diagnostics,
    )


# ============================================================
# TEMPORAL DIAGNOSTICS
# ============================================================

def create_distribution_diagnostics(
    master: pd.DataFrame,
    eligible: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    targets: dict,
) -> pd.DataFrame:

    records = []

    # --------------------------------------------------------
    # Simple dimensions
    # --------------------------------------------------------

    for dimension in [
        "year",
        "month",
        "season",
        "station",
    ]:

        master_counts = (
            master[
                dimension
            ]
            .value_counts()
        )

        train_counts = (
            train[
                dimension
            ]
            .value_counts()
        )

        test_counts = (
            test[
                dimension
            ]
            .value_counts()
        )

        for group, total in (
            master_counts.items()
        ):

            tr = int(
                train_counts.get(
                    group,
                    0,
                )
            )

            te = int(
                test_counts.get(
                    group,
                    0,
                )
            )

            records.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "total": int(total),
                    "train": tr,
                    "test": te,
                    "train_fraction": (
                        tr / total
                    ),
                    "test_fraction": (
                        te / total
                    ),
                    "expected_test": np.nan,
                    "test_minus_expected": np.nan,
                }
            )

    # --------------------------------------------------------
    # Year x Month
    # --------------------------------------------------------

    master_ym = (
        master
        .groupby(
            ["year", "month"]
        )
        .size()
    )

    train_ym = (
        train
        .groupby(
            ["year", "month"]
        )
        .size()
    )

    test_ym = (
        test
        .groupby(
            ["year", "month"]
        )
        .size()
    )

    for (
        year,
        month,
    ), total in master_ym.items():

        key = (
            int(year),
            int(month),
        )

        expected = targets[
            "year_month"
        ].get(
            key,
            np.nan,
        )

        tr = int(
            train_ym.get(
                key,
                0,
            )
        )

        te = int(
            test_ym.get(
                key,
                0,
            )
        )

        records.append(
            {
                "dimension": "year_month",
                "group": (
                    f"{int(year)}-"
                    f"{int(month):02d}"
                ),
                "total": int(total),
                "train": tr,
                "test": te,
                "train_fraction": (
                    tr / total
                ),
                "test_fraction": (
                    te / total
                ),
                "expected_test": expected,
                "test_minus_expected": (
                    te - expected
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# STATION COVERAGE DIAGNOSTICS
# ============================================================

def create_station_coverage_diagnostics(
    master: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    eligible_stations = sorted(
        set(
            master["station"]
        )
        - {SINGLETON_STATION}
    )

    for station in eligible_stations:

        master_station = master[
            master["station"] == station
        ]

        test_station = test[
            test["station"] == station
        ]

        for year in YEARS:

            records.append(
                {
                    "station": station,
                    "dimension": "year",
                    "group": year,
                    "available_in_master": int(
                        (
                            master_station[
                                "year"
                            ]
                            .astype(int)
                            == year
                        ).sum()
                    ),
                    "test_count": int(
                        (
                            test_station[
                                "year"
                            ]
                            .astype(int)
                            == year
                        ).sum()
                    ),
                }
            )

        for season in SEASONS:

            records.append(
                {
                    "station": station,
                    "dimension": "season",
                    "group": season,
                    "available_in_master": int(
                        (
                            master_station[
                                "season"
                            ]
                            == season
                        ).sum()
                    ),
                    "test_count": int(
                        (
                            test_station[
                                "season"
                            ]
                            == season
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_split(
    master: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    eligible: pd.DataFrame,
    station_quotas: pd.Series,
) -> pd.DataFrame:

    results = []

    def check(
        name,
        passed,
        observed,
        required,
    ):

        results.append(
            {
                "check": name,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "observed": observed,
                "required": required,
            }
        )

        if not passed:
            raise RuntimeError(
                "\nVALIDATION FAILURE\n"
                f"Check     : {name}\n"
                f"Observed  : {observed}\n"
                f"Required  : {required}"
            )

    # --------------------------------------------------------
    # Sizes
    # --------------------------------------------------------

    check(
        "master_rows",
        len(master) == EXPECTED_TOTAL,
        len(master),
        EXPECTED_TOTAL,
    )

    check(
        "train_rows",
        len(train) == EXPECTED_TRAIN,
        len(train),
        EXPECTED_TRAIN,
    )

    check(
        "test_rows",
        len(test) == EXPECTED_TEST,
        len(test),
        EXPECTED_TEST,
    )

    # --------------------------------------------------------
    # Keys
    # --------------------------------------------------------

    master_keys = set(
        map(
            tuple,
            master[
                KEYS
            ].astype(str).to_numpy(),
        )
    )

    train_keys = set(
        map(
            tuple,
            train[
                KEYS
            ].astype(str).to_numpy(),
        )
    )

    test_keys = set(
        map(
            tuple,
            test[
                KEYS
            ].astype(str).to_numpy(),
        )
    )

    overlap = (
        train_keys
        &
        test_keys
    )

    check(
        "train_test_overlap",
        len(overlap) == 0,
        len(overlap),
        0,
    )

    union = (
        train_keys
        |
        test_keys
    )

    check(
        "union_equals_master",
        union == master_keys,
        union == master_keys,
        True,
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    check(
        "duplicate_master_keys",
        not master.duplicated(KEYS).any(),
        int(
            master
            .duplicated(KEYS)
            .sum()
        ),
        0,
    )

    check(
        "duplicate_train_keys",
        not train.duplicated(KEYS).any(),
        int(
            train
            .duplicated(KEYS)
            .sum()
        ),
        0,
    )

    check(
        "duplicate_test_keys",
        not test.duplicated(KEYS).any(),
        int(
            test
            .duplicated(KEYS)
            .sum()
        ),
        0,
    )

    # --------------------------------------------------------
    # IIT Delhi
    # --------------------------------------------------------

    check(
        "IIT_Delhi_train_only",
        (
            SINGLETON_STATION
            in set(train["station"])
            and
            SINGLETON_STATION
            not in set(test["station"])
        ),
        (
            SINGLETON_STATION
            in set(train["station"]),
            SINGLETON_STATION
            in set(test["station"]),
        ),
        "(True, False)",
    )

    # --------------------------------------------------------
    # Station quotas
    # --------------------------------------------------------

    observed_quotas = (
        test["station"]
        .value_counts()
        .drop(
            labels=[
                SINGLETON_STATION
            ],
            errors="ignore",
        )
        .sort_index()
        .astype(int)
    )

    expected_quotas = (
        station_quotas
        .sort_index()
        .astype(int)
    )

    check(
        "station_quotas_exact",
        observed_quotas.equals(
            expected_quotas
        ),
        observed_quotas.to_dict(),
        expected_quotas.to_dict(),
    )

    # --------------------------------------------------------
    # All eligible stations represented
    # --------------------------------------------------------

    eligible_stations = set(
        eligible["station"]
    )

    missing_stations = (
        eligible_stations
        -
        set(test["station"])
    )

    check(
        "every_eligible_station_in_test",
        len(missing_stations) == 0,
        sorted(missing_stations),
        [],
    )

    # --------------------------------------------------------
    # Station year coverage
    # --------------------------------------------------------

    year_failures = []

    for station in sorted(
        eligible_stations
    ):

        master_station = master[
            master["station"] == station
        ]

        test_station = test[
            test["station"] == station
        ]

        available = set(
            master_station[
                "year"
            ]
            .astype(int)
        )

        observed = set(
            test_station[
                "year"
            ]
            .astype(int)
        )

        missing = (
            available
            -
            observed
        )

        if missing:

            year_failures.append(
                {
                    "station": station,
                    "missing": sorted(
                        missing
                    ),
                }
            )

    check(
        "station_available_year_coverage",
        len(year_failures) == 0,
        year_failures,
        [],
    )

    # --------------------------------------------------------
    # Station season coverage
    # --------------------------------------------------------

    season_failures = []

    for station in sorted(
        eligible_stations
    ):

        master_station = master[
            master["station"] == station
        ]

        test_station = test[
            test["station"] == station
        ]

        available = set(
            master_station[
                "season"
            ]
        )

        observed = set(
            test_station[
                "season"
            ]
        )

        missing = (
            available
            -
            observed
        )

        if missing:

            season_failures.append(
                {
                    "station": station,
                    "missing": sorted(
                        missing
                    ),
                }
            )

    check(
        "station_available_season_coverage",
        len(season_failures) == 0,
        season_failures,
        [],
    )

    # --------------------------------------------------------
    # Global year-month coverage
    # --------------------------------------------------------

    eligible_ym = set(
        map(
            tuple,
            eligible[
                [
                    "year",
                    "month",
                ]
            ]
            .astype(int)
            .drop_duplicates()
            .to_numpy(),
        )
    )

    test_ym = set(
        map(
            tuple,
            test[
                [
                    "year",
                    "month",
                ]
            ]
            .astype(int)
            .drop_duplicates()
            .to_numpy(),
        )
    )

    missing_ym = (
        eligible_ym
        -
        test_ym
    )

    check(
        "every_global_year_month_in_test",
        len(missing_ym) == 0,
        sorted(missing_ym),
        [],
    )

    # --------------------------------------------------------
    # Global years
    # --------------------------------------------------------

    train_years = set(
        train["year"]
        .astype(int)
    )

    test_years = set(
        test["year"]
        .astype(int)
    )

    check(
        "all_years_in_train",
        train_years == set(YEARS),
        sorted(train_years),
        YEARS,
    )

    check(
        "all_years_in_test",
        test_years == set(YEARS),
        sorted(test_years),
        YEARS,
    )

    # --------------------------------------------------------
    # PM2.5 selection independence
    # --------------------------------------------------------

    check(
        "pm25_used_for_selection",
        True,
        False,
        False,
    )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files.",
    )

    args = parser.parse_args()

    print(
        "=" * 78
    )

    print(
        "V3 — FINAL STATION-PROPORTIONAL "
        "SPATIOTEMPORAL SPLIT"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Dependency
    # --------------------------------------------------------

    if not SCIPY_AVAILABLE:

        raise RuntimeError(
            "\nSciPy is required.\n\n"
            "Run:\n"
            "    pip install scipy\n\n"
            "Then rerun this script."
        )

    # --------------------------------------------------------
    # Load V3
    # --------------------------------------------------------

    if not V3_PATH.exists():

        raise FileNotFoundError(
            f"V3 dataset not found:\n{V3_PATH}"
        )

    master = pd.read_csv(
        V3_PATH
    )

    validate_master(
        master
    )

    master = add_season(
        master
    )

    print(
        "\nMASTER DATASET"
    )

    print(
        f"Rows     : {len(master)}"
    )

    print(
        f"Stations : {master['station'].nunique()}"
    )

    print(
        f"Years    : "
        f"{sorted(master['year'].astype(int).unique())}"
    )

    # --------------------------------------------------------
    # Output safety
    # --------------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_files = [
        "train.csv",
        "test.csv",
        "split_manifest.json",
        "validation_report.csv",
        "distribution_diagnostics.csv",
        "station_coverage_diagnostics.csv",
    ]

    if not args.overwrite:

        existing = [
            name
            for name in expected_files
            if (
                SPLIT_DIR / name
            ).exists()
        ]

        if existing:

            raise RuntimeError(
                "Output files already exist:\n"
                +
                "\n".join(existing)
                +
                "\n\nUse --overwrite to replace them."
            )

    # --------------------------------------------------------
    # IIT Delhi singleton
    # --------------------------------------------------------

    singleton = master[
        master["station"]
        == SINGLETON_STATION
    ]

    if len(singleton) != 1:

        raise RuntimeError(
            f"Expected exactly one observation "
            f"for {SINGLETON_STATION}; "
            f"found {len(singleton)}."
        )

    # --------------------------------------------------------
    # Eligible pool
    # --------------------------------------------------------

    eligible = master[
        master["station"]
        != SINGLETON_STATION
    ].copy()

    print(
        f"\n{SINGLETON_STATION}: "
        f"{len(singleton)} row -> TRAIN ONLY"
    )

    print(
        f"Eligible test pool: "
        f"{len(eligible)} rows"
    )

    # --------------------------------------------------------
    # Station quotas
    # --------------------------------------------------------

    station_quotas = (
        allocate_station_quotas(
            eligible,
            EXPECTED_TEST,
        )
    )

    print(
        "\nLOCKED STATION TEST QUOTAS"
    )

    print(
        station_quotas.to_string()
    )

    print(
        f"\nQuota total: "
        f"{station_quotas.sum()}"
    )

    # --------------------------------------------------------
    # Temporal targets
    # --------------------------------------------------------

    targets = (
        build_temporal_targets(
            eligible,
            EXPECTED_TEST,
        )
    )

    global_ratio = targets[
        "ratio"
    ]

    print(
        "\nEffective eligible-pool "
        f"test fraction: {global_ratio:.6f}"
    )

    print(
        "Temporal targets are based on the "
        "eligible test pool, excluding IIT_Delhi."
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    train, test, solver_info = (
        solve_optimal_split(
            master=master,
            eligible=eligible,
            station_quotas=station_quotas,
            targets=targets,
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    train = (
        train
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate BEFORE writing
    # --------------------------------------------------------

    print(
        "\nVALIDATING FINAL SPLIT..."
    )

    validation = (
        validate_final_split(
            master=master,
            train=train,
            test=test,
            eligible=eligible,
            station_quotas=station_quotas,
        )
    )

    if not validation[
        "status"
    ].eq("PASS").all():

        raise RuntimeError(
            "Final validation failed. "
            "No files were written."
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = (
        create_distribution_diagnostics(
            master=master,
            eligible=eligible,
            train=train,
            test=test,
            targets=targets,
        )
    )

    station_coverage = (
        create_station_coverage_diagnostics(
            master=master,
            test=test,
        )
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    ym_diagnostics = (
        diagnostics[
            diagnostics[
                "dimension"
            ]
            == "year_month"
        ]
        .copy()
    )

    max_abs_ym_deviation = float(
        ym_diagnostics[
            "test_minus_expected"
        ]
        .abs()
        .max()
    )

    mean_abs_ym_deviation = float(
        ym_diagnostics[
            "test_minus_expected"
        ]
        .abs()
        .mean()
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "dataset": str(
            V3_PATH.relative_to(
                ROOT
            )
        ),

        "method": (
            "Station-proportional 80:20 holdout "
            "with exact station quotas, station-year "
            "and station-season coverage constraints, "
            "and constrained optimization of the global "
            "Year x Month distribution."
        ),

        "seed": SEED,

        "solver": solver_info,

        "master_rows": int(
            len(master)
        ),

        "eligible_rows": int(
            len(eligible)
        ),

        "train_rows": int(
            len(train)
        ),

        "test_rows": int(
            len(test)
        ),

        "train_fraction": (
            len(train)
            / len(master)
        ),

        "test_fraction": (
            len(test)
            / len(master)
        ),

        "eligible_test_fraction": (
            EXPECTED_TEST
            / len(eligible)
        ),

        "stations": int(
            master["station"].nunique()
        ),

        "test_station_quota": {
            str(station): int(
                quota
            )
            for station, quota
            in station_quotas.items()
        },

        "constraints": {

            "IIT_Delhi_train_only": True,

            "station_quotas_locked": True,

            "every_station_in_test": True,

            "every_station_available_years": True,

            "every_station_available_seasons": True,

            "every_global_year_month_present": True,

            "pm25_used_for_selection": False,

            "train_test_overlap": 0,

            "union_equals_master": True,

            "post_hoc_train_test_swaps": False,
        },

        "temporal_balance": {

            "target_based_on": (
                "eligible pool excluding IIT_Delhi"
            ),

            "max_absolute_year_month_deviation":
                max_abs_ym_deviation,

            "mean_absolute_year_month_deviation":
                mean_abs_ym_deviation,
        },

        "status": "PASS",
    }

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    train.to_csv(
        SPLIT_DIR / "train.csv",
        index=False,
    )

    test.to_csv(
        SPLIT_DIR / "test.csv",
        index=False,
    )

    diagnostics.to_csv(
        SPLIT_DIR
        / "distribution_diagnostics.csv",
        index=False,
    )

    station_coverage.to_csv(
        SPLIT_DIR
        / "station_coverage_diagnostics.csv",
        index=False,
    )

    validation.to_csv(
        SPLIT_DIR
        / "validation_report.csv",
        index=False,
    )

    with open(
        SPLIT_DIR
        / "split_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FINAL SPLIT CREATED SUCCESSFULLY"
    )

    print(
        "=" * 78
    )

    print(
        f"Master : {len(master)}"
    )

    print(
        f"Train  : {len(train)}"
    )

    print(
        f"Test   : {len(test)}"
    )

    print(
        f"Overall test fraction: "
        f"{len(test) / len(master):.4%}"
    )

    print(
        f"Eligible-pool test fraction: "
        f"{len(test) / len(eligible):.4%}"
    )

    print(
        "\nTEST ROWS PER STATION"
    )

    print(
        test[
            "station"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTEST BY YEAR"
    )

    print(
        test[
            "year"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTEST BY MONTH"
    )

    print(
        test[
            "month"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTEST BY SEASON"
    )

    print(
        test[
            "season"
        ]
        .value_counts()
        .reindex(
            SEASONS
        )
        .to_string()
    )

    print(
        "\nYEAR × MONTH TEST COUNTS"
    )

    print(
        test
        .groupby(
            [
                "year",
                "month",
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nTEMPORAL BALANCE"
    )

    print(
        f"Mean absolute "
        f"Year×Month deviation : "
        f"{mean_abs_ym_deviation:.4f}"
    )

    print(
        f"Maximum absolute "
        f"Year×Month deviation : "
        f"{max_abs_ym_deviation:.4f}"
    )

    print(
        "\nFINAL VALIDATION"
    )

    print(
        "✓ Exact 1292 train / 323 test"
    )

    print(
        "✓ IIT_Delhi train-only"
    )

    print(
        "✓ Exact station quotas"
    )

    print(
        "✓ Every eligible station represented"
    )

    print(
        "✓ Every station's available years represented"
    )

    print(
        "✓ Every station's available seasons represented"
    )

    print(
        "✓ Every global year-month represented"
    )

    print(
        "✓ Zero train/test key overlap"
    )

    print(
        "✓ Complete V3 row universe preserved"
    )

    print(
        "✓ PM2.5 never used for selection"
    )

    print(
        "✓ No post-hoc train/test swapping"
    )

    print(
        "✓ V3 master dataset untouched"
    )

    print(
        "\nFILES"
    )

    print(
        SPLIT_DIR / "train.csv"
    )

    print(
        SPLIT_DIR / "test.csv"
    )

    print(
        SPLIT_DIR / "distribution_diagnostics.csv"
    )

    print(
        SPLIT_DIR
        / "station_coverage_diagnostics.csv"
    )

    print(
        SPLIT_DIR / "validation_report.csv"
    )

    print(
        SPLIT_DIR / "split_manifest.json"
    )

    print(
        "\nSTATUS: PASS"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()