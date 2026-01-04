from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


# 0. Configuration
RANDOM_STATE = 42

COLS_X_RAW = [
    # DC-link voltage
    "u_dc_k", "u_dc_k-1", "u_dc_k-2", "u_dc_k-3",

    # Phase currents
    "i_a_k", "i_b_k", "i_c_k",
    "i_a_k-1", "i_b_k-1", "i_c_k-1",
    "i_a_k-2", "i_b_k-2", "i_c_k-2",
    "i_a_k-3", "i_b_k-3", "i_c_k-3",

    # Duty cycles
    "d_a_k-2", "d_b_k-2", "d_c_k-2",
    "d_a_k-3", "d_b_k-3", "d_c_k-3",
]


# Targets
COLS_Y = ["u_a_k-1", "u_b_k-1", "u_c_k-1"]


# 1. Load data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Provide the correct path to your CSV"
        )

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Supported formats: .csv")

    df = df.drop(columns=["n_k"], errors="ignore")

    # Basic checks
    missing_x = [c for c in COLS_X_RAW if c not in df.columns]

    missing_y = [c for c in COLS_Y if c not in df.columns]
    if missing_x or missing_y:
        raise ValueError(
            "Missing required columns.\n"
            f"Missing X: {missing_x}\n"
            f"Missing Y: {missing_y}\n"
            f"Available columns: {list(df.columns)[:50]}..."
        )

    # Drop rows with missing values in required cols
    df = df.dropna(subset=COLS_X_RAW + COLS_Y).reset_index(drop=True)

    # Use float32 to reduce memory footprint
    for c in COLS_X_RAW + COLS_Y:
        df[c] = df[c].astype(np.float32)

    return df


# 2. Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Simple deltas
    out["d_udc"] = out["u_dc_k"] - out["u_dc_k-1"]

    out["d_ia"] = out["i_a_k"] - out["i_a_k-1"]
    out["d_ib"] = out["i_b_k"] - out["i_b_k-1"]
    out["d_ic"] = out["i_c_k"] - out["i_c_k-1"]

    out["d_da"] = out["d_a_k-2"] - out["d_a_k-3"]
    out["d_db"] = out["d_b_k-2"] - out["d_b_k-3"]
    out["d_dc"] = out["d_c_k-2"] - out["d_c_k-3"]

    # Current sums (balance check)
    out["isum_k"] = out["i_a_k"] + out["i_b_k"] + out["i_c_k"]
    out["isum_k_1"] = out["i_a_k-1"] + out["i_b_k-1"] + out["i_c_k-1"]

    # Duty averages
    out["da_avg"] = 0.5 * (out["d_a_k-2"] + out["d_a_k-3"])
    out["db_avg"] = 0.5 * (out["d_b_k-2"] + out["d_b_k-3"])
    out["dc_avg"] = 0.5 * (out["d_c_k-2"] + out["d_c_k-3"])

    # Voltage-duty proxy
    out["udc1_da"] = out["u_dc_k-1"] * out["da_avg"]
    out["udc1_db"] = out["u_dc_k-1"] * out["db_avg"]
    out["udc1_dc"] = out["u_dc_k-1"] * out["dc_avg"]

    return out



def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # Use all engineered columns + raw inputs
    engineered = [
        "d_udc", "d_ia", "d_ib", "d_ic",
        "d_da", "d_db", "d_dc",
        "isum_k", "isum_k_1",
        "da_avg", "db_avg", "dc_avg",
        "udc1_da", "udc1_db", "udc1_dc",
    ]
    cols = COLS_X_RAW + engineered
    # Ensure all exist
    cols = [c for c in cols if c in df.columns]
    return cols


# 3. Metrics + plots
@dataclass
class EvalResult:
    mae: np.ndarray
    rmse: np.ndarray
    r2: np.ndarray

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    # per-output metrics
    mae = np.array([mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    rmse = np.array([np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))for i in range(y_true.shape[1])])

    r2 = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    return EvalResult(mae=mae, rmse=rmse, r2=r2)

def print_eval(name: str, res: EvalResult) -> None:
    print(f"\n=== {name} ===")
    for i, phase in enumerate(["a", "b", "c"]):
        print(f"Phase {phase}: MAE={res.mae[i]:.4f}, RMSE={res.rmse[i]:.4f}, R2={res.r2[i]:.4f}")
    print(f"Avg:     MAE={res.mae.mean():.4f}, RMSE={res.rmse.mean():.4f}, R2={res.r2.mean():.4f}")

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    phases = ["a", "b", "c"]
    for i, ph in enumerate(phases):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], s=4, alpha=0.25)
        mn = float(min(y_true[:, i].min(), y_pred[:, i].min()))
        mx = float(max(y_true[:, i].max(), y_pred[:, i].max()))
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel(f"True u{ph},k-1")
        plt.ylabel(f"Pred u{ph},k-1")
        plt.title(f"{title} - Pred vs True (phase {ph})")
        plt.tight_layout()
        plt.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    phases = ["a", "b", "c"]
    resid = y_true - y_pred
    for i, ph in enumerate(phases):
        plt.figure()
        plt.hist(resid[:, i], bins=80)
        plt.xlabel(f"Residual (u{ph},k-1 true - pred)")
        plt.ylabel("Count")
        plt.title(f"{title} - Residual histogram (phase {ph})")
        plt.tight_layout()
        plt.show()


# 4. Model builders
def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    # All numeric -> scale (especially important for Ridge and MLP)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols)
        ],
        remainder="drop"
    )

def make_ridge_model(feature_cols: List[str]) -> Pipeline:
    pre = make_preprocessor(feature_cols)
    model = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    # Ridge supports multioutput directly if y is (n,3)
    return Pipeline([("pre", pre), ("model", model)])

def make_hgbr_model(feature_cols: List[str]) -> Pipeline:
    pre = make_preprocessor(feature_cols)
    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=8,
        max_iter=400,
        min_samples_leaf=40,
        l2_regularization=1e-3,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    # Wrap for multi-output
    model = MultiOutputRegressor(base, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])

def make_mlp_model(feature_cols: List[str]) -> Pipeline:
    pre = make_preprocessor(feature_cols)
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        alpha=1e-4,                 # L2
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    return Pipeline([("pre", pre), ("model", model)])


# 5. Hyperparameter tuning
def tune_hgbr(X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: List[str]) -> Pipeline:
    pre = make_preprocessor(feature_cols)
    base = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("pre", pre), ("model", MultiOutputRegressor(base, n_jobs=-1))])

    # Note: params must be addressed through the nested estimator inside MultiOutputRegressor
    param_dist = {
        "model__estimator__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__estimator__max_depth": [4, 6, 8, 10],
        "model__estimator__max_iter": [200, 400, 800],
        "model__estimator__min_samples_leaf": [20, 40, 80, 120],
        "model__estimator__l2_regularization": [0.0, 1e-4, 1e-3, 1e-2],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train[feature_cols], y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_


# 6. Main
def main(dataset_path: str, do_tuning: bool = False) -> None:
    df = load_data(dataset_path)
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df[COLS_Y].to_numpy()

    # Split: train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    # 1) Ridge baseline
    ridge = make_ridge_model(feature_cols)
    ridge.fit(X_train, y_train)
    pred_train = ridge.predict(X_train)
    pred_val = ridge.predict(X_val)
    pred_test = ridge.predict(X_test)
    print_eval("Ridge (TRAIN)", evaluate(y_train, pred_train))
    print_eval("Ridge (VAL)", evaluate(y_val, pred_val))
    print_eval("Ridge (TEST)", evaluate(y_test, pred_test))

    # 2) HistGradientBoosting (or tuned variant)
    if do_tuning:
        hgbr = tune_hgbr(X_train, y_train, feature_cols)
    else:
        hgbr = make_hgbr_model(feature_cols)

    hgbr.fit(X_train, y_train)
    pred_train = hgbr.predict(X_train)
    pred_val = hgbr.predict(X_val)
    pred_test = hgbr.predict(X_test)
    print_eval("HGBR (TRAIN)", evaluate(y_train, pred_train))
    print_eval("HGBR (VAL)", evaluate(y_val, pred_val))
    print_eval("HGBR (TEST)", evaluate(y_test, pred_test))

    # 3) MLP
    mlp = make_mlp_model(feature_cols)
    mlp.fit(X_train, y_train)
    pred_train = mlp.predict(X_train)
    pred_val = mlp.predict(X_val)
    pred_test = mlp.predict(X_test)
    print_eval("MLP (TRAIN)", evaluate(y_train, pred_train))
    print_eval("MLP (VAL)", evaluate(y_val, pred_val))
    print_eval("MLP (TEST)", evaluate(y_test, pred_test))

    # Choose best model and plot diagnostics on TEST
    best_model = hgbr
    best_pred = best_model.predict(X_test)
    plot_pred_vs_true(y_test, best_pred, title="Best model (TEST)")
    plot_residuals(y_test, best_pred, title="Best model (TEST)")


if __name__ == "__main__":

    main("data.csv", do_tuning=False)
