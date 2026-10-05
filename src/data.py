"""
Data loading + preprocessing for FedActuary. Verified against the base
paper's own reference implementation (github.com/actuari/IFoA-FL-WP),
including its actual split_data() function -- PRD Section 5.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

AREA_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
SCALE_COLS = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
EXPECTED_FEATURES = 39

# Canonical categorical levels matching freMTPL2freq.csv
VEH_BRANDS = ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"]
REGIONS = [
    "R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31",
    "R41", "R42", "R43", "R52", "R53", "R54", "R72", "R73",
    "R74", "R82", "R83", "R91", "R93", "R94"
]


def load_raw_features(csv_path: str, n_rows: int = None, seed: int = 42, return_metadata: bool = False):
    """
    Loads raw freMTPL2freq data and performs deterministic preprocessing:
    1. ClaimNb -> clip(upper=4)
    2. VehAge -> clip(upper=20)
    3. DrivAge -> clip(upper=90)
    4. BonusMalus -> clip(upper=150)
    5. Density -> log-transform, before scaling
    6. Exposure -> clip(upper=1)
    7. Drop IDpol
    8. Area -> map A-F to 1-6
    9. VehGas -> map {Regular: 1, Diesel: 2}
    10. Categorical dummies for VehBrand, Region with drop_first=True
    """
    df = pd.read_csv(csv_path)
    if n_rows is not None:
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

    # Save raw metadata for Dirichlet partitioning if requested
    raw_regions = df["Region"].astype(str).values if "Region" in df.columns else None

    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"], errors="coerce").clip(upper=4)
    df["VehAge"] = df["VehAge"].clip(upper=20)
    df["DrivAge"] = df["DrivAge"].clip(upper=90)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=150)
    df["Density"] = np.log(df["Density"].clip(lower=1e-6))
    df["Exposure"] = df["Exposure"].clip(upper=1)

    if "IDpol" in df.columns:
        df = df.drop(columns=["IDpol"])

    df["Area"] = df["Area"].map(AREA_MAP)
    df["VehGas"] = df["VehGas"].map({"Regular": 1, "Diesel": 2})

    # Set canonical categories to guarantee exact 39 columns regardless of sample size
    df["VehBrand"] = pd.Categorical(df["VehBrand"], categories=VEH_BRANDS)
    df["Region"] = pd.Categorical(df["Region"], categories=REGIONS)

    df = pd.get_dummies(df, columns=["VehBrand", "Region"], drop_first=True)

    exposure = df["Exposure"].values.astype("float32")
    y = df["ClaimNb"].values.astype("float32")
    X_df = df.drop(columns=["ClaimNb"])

    if return_metadata:
        metadata = {"Region": raw_regions}
        return X_df, y, exposure, metadata
    return X_df, y, exposure


def split_and_scale(
    X_df,
    y,
    exposure,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    return_indices: bool = False
):
    """
    Splits into train (approx 72%), val (approx 8%), and test (approx 20%).
    Fits MinMaxScaler on training split only and transforms val and test.
    Optionally returns split indices to ensure exact alignment for Phase 2.
    """
    n = len(X_df)
    idx_trainval, idx_test = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_size, random_state=seed)

    X_train = X_df.iloc[idx_train].copy()
    X_val = X_df.iloc[idx_val].copy()
    X_test = X_df.iloc[idx_test].copy()

    scaler = MinMaxScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_val[SCALE_COLS] = scaler.transform(X_val[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    feature_names = list(X_train.columns)
    X_train_arr = X_train.astype("float32").values
    X_val_arr = X_val.astype("float32").values
    X_test_arr = X_test.astype("float32").values

    assert X_train_arr.shape[1] == EXPECTED_FEATURES, (
        f"expected {EXPECTED_FEATURES} features, got {X_train_arr.shape[1]}"
    )

    train = (X_train_arr, y[idx_train], exposure[idx_train])
    val = (X_val_arr, y[idx_val], exposure[idx_val])
    test = (X_test_arr, y[idx_test], exposure[idx_test])

    if return_indices:
        indices = {"train": idx_train, "val": idx_val, "test": idx_test}
        return train, val, test, scaler, feature_names, indices
    return train, val, test, scaler, feature_names


def preprocess_single_record(record: dict, scaler: MinMaxScaler, feature_names: list) -> np.ndarray:
    """
    Transforms a single policy record dictionary (with the 12 raw fields)
    into a (1, 39) numpy array ready for inference, using the persisted scaler.
    Per PRD Section 8.1 & Section 12 item 10, does NOT refit the scaler.
    """
    df = pd.DataFrame([record])

    # Preprocessing identical to load_raw_features
    df["VehAge"] = pd.to_numeric(df.get("VehAge", 0), errors="coerce").fillna(0).clip(upper=20)
    df["DrivAge"] = pd.to_numeric(df.get("DrivAge", 40), errors="coerce").fillna(40).clip(upper=90)
    df["BonusMalus"] = pd.to_numeric(df.get("BonusMalus", 100), errors="coerce").fillna(100).clip(upper=150)
    density_val = pd.to_numeric(df.get("Density", 100), errors="coerce").fillna(100)
    df["Density"] = np.log(np.maximum(density_val, 1e-6))
    df["Exposure"] = pd.to_numeric(df.get("Exposure", 1.0), errors="coerce").fillna(1.0).clip(upper=1.0)
    df["VehPower"] = pd.to_numeric(df.get("VehPower", 6), errors="coerce").fillna(6)

    area_val = str(df.get("Area", ["C"])[0]).upper()
    df["Area"] = AREA_MAP.get(area_val, 3)

    gas_val = str(df.get("VehGas", ["Regular"])[0])
    df["VehGas"] = 1 if gas_val.lower() == "regular" else 2

    df["VehBrand"] = pd.Categorical(df.get("VehBrand", ["B1"]), categories=VEH_BRANDS)
    df["Region"] = pd.Categorical(df.get("Region", ["R11"]), categories=REGIONS)

    df_dummies = pd.get_dummies(df[["VehBrand", "Region"]], drop_first=True)
    numeric_cols = ["Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehGas", "Density"]
    full_df = pd.concat([df[numeric_cols], df_dummies], axis=1)

    # Scale the 6 numerical features using the fitted scaler
    full_df[SCALE_COLS] = scaler.transform(full_df[SCALE_COLS])

    # Reorder columns to match exactly the 39 features
    full_df = full_df.reindex(columns=feature_names, fill_value=0.0)
    return full_df.astype("float32").values


def split_for_clients(X, y, exposure, num_clients: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    shards = np.array_split(idx, num_clients)
    return [(X[s], y[s], exposure[s]) for s in shards]


def save_preprocessing_artifacts(scaler, feature_names, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f)


def load_preprocessing_artifacts(out_dir: str):
    with open(os.path.join(out_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(out_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return scaler, feature_names
