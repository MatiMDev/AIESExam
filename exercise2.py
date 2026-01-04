from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 0. Reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 1. Analytic solution (underdamped)
def damped_oscillator_solution(
    t: np.ndarray,
    m: float,
    c: float,
    k: float,
    u0: float,
    v0: float,
) -> np.ndarray:
    omega_n = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(m * k))
    assert zeta < 1.0, "This analytic formula assumes underdamped (zeta < 1)."

    omega_d = omega_n * math.sqrt(1.0 - zeta**2)
    # u(t) = e^{-zeta omega_n t} [ A cos(omega_d t) + B sin(omega_d t) ]
    A = u0
    # v(0) = -zeta*omega_n*A + omega_d*B  =>  B = (v0 + zeta*omega_n*A)/omega_d
    B = (v0 + zeta * omega_n * A) / omega_d

    return np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))


# 2. Data generation
@dataclass
class DataConfig:
    m: float = 1.0
    c: float = 0.35
    k: float = 4.0
    u0: float = 1.0
    v0: float = 0.0

    t_min: float = 0.0
    t_max: float = 10.0

    n_train: int = 25               # sparse data to highlight generalization issues
    n_val: int = 25
    n_test: int = 400               # dense evaluation grid
    noise_std: float = 0.02         # measurement noise std


def make_dataset(cfg: DataConfig) -> Dict[str, np.ndarray]:
    t_test = np.linspace(cfg.t_min, cfg.t_max, cfg.n_test)
    u_test_true = damped_oscillator_solution(t_test, cfg.m, cfg.c, cfg.k, cfg.u0, cfg.v0)

    # Training/validation: sample only in a limited window to show extrapolation weakness
    t_train = np.sort(np.random.uniform(cfg.t_min, 3.0, size=cfg.n_train))
    t_val = np.sort(np.random.uniform(cfg.t_min, 3.0, size=cfg.n_val))

    u_train_true = damped_oscillator_solution(t_train, cfg.m, cfg.c, cfg.k, cfg.u0, cfg.v0)
    u_val_true = damped_oscillator_solution(t_val, cfg.m, cfg.c, cfg.k, cfg.u0, cfg.v0)

    u_train = u_train_true + np.random.normal(0.0, cfg.noise_std, size=cfg.n_train)
    u_val = u_val_true + np.random.normal(0.0, cfg.noise_std, size=cfg.n_val)

    return {
        "t_train": t_train, "u_train": u_train, "u_train_true": u_train_true,
        "t_val": t_val, "u_val": u_val, "u_val_true": u_val_true,
        "t_test": t_test, "u_test_true": u_test_true,
    }


# 3. Model
class MLP(nn.Module):
    def __init__(self, layers: List[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        mods: List[nn.Module] = []
        for i in range(len(layers) - 2):
            mods.append(nn.Linear(layers[i], layers[i+1]))
            mods.append(activation)
        mods.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)

        # Xavier init is typically stable for tanh MLPs
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 4. Metrics
def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))

def mae(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(y_hat - y)))

def r2(y_hat: np.ndarray, y: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# 5. Training loops
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 6000
    patience: int = 400
    print_every: int = 500

    # collocation points for physics term
    n_collocation: int = 200
    lambda_phys: float = 1.0


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)


def train_conventional_nn(
    model: nn.Module,
    t_train: np.ndarray,
    u_train: np.ndarray,
    t_val: np.ndarray,
    u_val: np.ndarray,
    cfg: TrainConfig
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = to_tensor(t_train, cfg.device)
    Ytr = to_tensor(u_train, cfg.device)
    Xva = to_tensor(t_val, cfg.device)
    Yva = to_tensor(u_val, cfg.device)

    best_state = None
    best_val = float("inf")
    bad = 0

    hist = {"train": [], "val": []}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()

        hist["train"].append(float(loss.item()))
        hist["val"].append(float(val_loss))

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % cfg.print_every == 0 or ep == 1:
            print(f"[NN] epoch={ep:5d} train={loss.item():.4e} val={val_loss:.4e} (best {best_val:.4e})")

        if bad >= cfg.patience:
            print(f"[NN] early stopping at epoch={ep} (best val {best_val:.4e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist


def pinn_physics_residual(
    model: nn.Module,
    t: torch.Tensor,
    m: float,
    c: float,
    k: float
) -> torch.Tensor:
    """
    Compute residual r(t) = m u_tt + c u_t + k u using autograd.
    """
    # ensure gradients w.r.t input
    t = t.clone().detach().requires_grad_(True)
    u = model(t)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    u_tt = torch.autograd.grad(
        u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True
    )[0]

    r = m * u_tt + c * u_t + k * u
    return r


def train_pinn(
    model: nn.Module,
    t_train: np.ndarray,
    u_train: np.ndarray,
    t_val: np.ndarray,
    u_val: np.ndarray,
    osc_params: Tuple[float, float, float],
    cfg: TrainConfig
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data_loss_fn = nn.MSELoss()

    m, c, k = osc_params

    Xtr = to_tensor(t_train, cfg.device)
    Ytr = to_tensor(u_train, cfg.device)
    Xva = to_tensor(t_val, cfg.device)
    Yva = to_tensor(u_val, cfg.device)

    # collocation points: sample across the full domain to enforce physics everywhere
    t_col = np.random.uniform(np.min(t_val), np.max([np.max(t_val), 10.0]), size=cfg.n_collocation)
    t_col = np.random.uniform(0.0, 10.0, size=cfg.n_collocation)  # full domain
    Xcol = to_tensor(t_col, cfg.device)

    best_state = None
    best_val = float("inf")
    bad = 0

    hist = {"data": [], "phys": [], "total": [], "val": []}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        # data loss
        u_hat = model(Xtr)
        loss_data = data_loss_fn(u_hat, Ytr)

        # physics loss
        r = pinn_physics_residual(model, Xcol, m, c, k)
        loss_phys = torch.mean(r**2)

        loss_total = loss_data + cfg.lambda_phys * loss_phys
        loss_total.backward()
        opt.step()

        # validation: compare to validation data only
        model.eval()
        with torch.no_grad():
            val_loss = data_loss_fn(model(Xva), Yva).item()

        hist["data"].append(float(loss_data.item()))
        hist["phys"].append(float(loss_phys.item()))
        hist["total"].append(float(loss_total.item()))
        hist["val"].append(float(val_loss))

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % cfg.print_every == 0 or ep == 1:
            print(
                f"[PINN] epoch={ep:5d} data={loss_data.item():.3e} phys={loss_phys.item():.3e} "
                f"total={loss_total.item():.3e} val={val_loss:.3e} (best {best_val:.3e})"
            )

        if bad >= cfg.patience:
            print(f"[PINN] early stopping at epoch={ep} (best val {best_val:.4e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist


# 6. Plotting helpers
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_fit(
    fig_path: str,
    title: str,
    t_train: np.ndarray,
    u_train: np.ndarray,
    t_test: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
) -> None:
    plt.figure()
    plt.plot(t_test, u_true, linewidth=2, label="True")
    plt.scatter(t_train, u_train, s=30, label="Training data", zorder=3)
    plt.plot(t_test, u_pred, linewidth=2, label="Model prediction")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_training_curves(fig_path: str, nn_hist: Dict[str, List[float]], pinn_hist: Dict[str, List[float]]) -> None:
    plt.figure()
    plt.plot(nn_hist["train"], label="NN train (data)")
    plt.plot(nn_hist["val"], label="NN val (data)")
    plt.plot(pinn_hist["total"], label="PINN train (total)")
    plt.plot(pinn_hist["val"], label="PINN val (data)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training curves (note PINN typically converges slower)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

def plot_sensitivity(fig_path: str, lambdas: List[float], rmses: List[float]) -> None:
    plt.figure()
    plt.plot(lambdas, rmses, marker="o")
    plt.xscale("log")
    plt.xlabel(r"$\lambda_{\mathrm{phys}}$ (log scale)")
    plt.ylabel("RMSE on test grid")
    plt.title("Sensitivity to physics-loss weight")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# 7. Main experiment
def evaluate_model_on_grid(model: nn.Module, t_test: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = to_tensor(t_test, device)
        y = model(X).detach().cpu().numpy().reshape(-1)
    return y


def run_once(lambda_phys: float) -> Tuple[float, Dict[str, float]]:
    set_seed(42)

    data_cfg = DataConfig()
    ds = make_dataset(data_cfg)

    # baseline NN
    nn_model = MLP([1, 64, 64, 64, 1], activation=nn.Tanh())
    nn_cfg = TrainConfig(lambda_phys=0.0, epochs=5000, patience=350, lr=1e-3, print_every=1000)

    nn_model, nn_hist = train_conventional_nn(
        nn_model, ds["t_train"], ds["u_train"], ds["t_val"], ds["u_val"], nn_cfg
    )
    nn_pred = evaluate_model_on_grid(nn_model, ds["t_test"], nn_cfg.device)

    # PINN
    pinn_model = MLP([1, 64, 64, 64, 1], activation=nn.Tanh())
    pinn_cfg = TrainConfig(lambda_phys=lambda_phys, epochs=7000, patience=500, lr=1e-3, print_every=1000)
    pinn_model, pinn_hist = train_pinn(
        pinn_model, ds["t_train"], ds["u_train"], ds["t_val"], ds["u_val"],
        (data_cfg.m, data_cfg.c, data_cfg.k),
        pinn_cfg
    )
    pinn_pred = evaluate_model_on_grid(pinn_model, ds["t_test"], pinn_cfg.device)

    # Metrics vs true (dense grid)
    metrics = {
        "NN_RMSE": rmse(nn_pred, ds["u_test_true"]),
        "NN_MAE": mae(nn_pred, ds["u_test_true"]),
        "NN_R2": r2(nn_pred, ds["u_test_true"]),
        "PINN_RMSE": rmse(pinn_pred, ds["u_test_true"]),
        "PINN_MAE": mae(pinn_pred, ds["u_test_true"]),
        "PINN_R2": r2(pinn_pred, ds["u_test_true"]),
    }

    # Save plots for the “main” lambda setting
    ensure_dir("figs")
    if abs(lambda_phys - 1.0) < 1e-12:
        plot_fit("figs/nn_fit.png", "Conventional NN fit (data-driven)",
                 ds["t_train"], ds["u_train"], ds["t_test"], ds["u_test_true"], nn_pred)
        plot_fit("figs/pinn_fit.png", f"PINN fit (lambda_phys={lambda_phys})",
                 ds["t_train"], ds["u_train"], ds["t_test"], ds["u_test_true"], pinn_pred)
        plot_training_curves("figs/training_curves.png", nn_hist, pinn_hist)

    return metrics["PINN_RMSE"], metrics

# 8. Data volume study

def run_data_volume_study(
    train_sizes: List[int],
    lambda_phys: float = 1.0
) -> None:
    """
    Study effect of training data volume on NN vs PINN performance.
    This function is purely additive and does not modify existing experiments.
    """
    set_seed(42)

    nn_rmses = []
    pinn_rmses = []

    for n in train_sizes:
        print(f"\n--- Data volume study: n_train={n} ---")

        data_cfg = DataConfig(n_train=n, n_val=n)
        ds = make_dataset(data_cfg)

        # NN
        nn_model = MLP([1, 64, 64, 64, 1], activation=nn.Tanh())
        nn_cfg = TrainConfig(lambda_phys=0.0, epochs=5000, patience=350, lr=1e-3)

        nn_model, _ = train_conventional_nn(
            nn_model, ds["t_train"], ds["u_train"], ds["t_val"], ds["u_val"], nn_cfg
        )
        nn_pred = evaluate_model_on_grid(nn_model, ds["t_test"], nn_cfg.device)
        nn_rmse_val = rmse(nn_pred, ds["u_test_true"])
        nn_rmses.append(nn_rmse_val)

        # PINN
        pinn_model = MLP([1, 64, 64, 64, 1], activation=nn.Tanh())
        pinn_cfg = TrainConfig(lambda_phys=lambda_phys, epochs=7000, patience=500, lr=1e-3)

        pinn_model, _ = train_pinn(
            pinn_model,
            ds["t_train"], ds["u_train"],
            ds["t_val"], ds["u_val"],
            (data_cfg.m, data_cfg.c, data_cfg.k),
            pinn_cfg
        )
        pinn_pred = evaluate_model_on_grid(pinn_model, ds["t_test"], pinn_cfg.device)
        pinn_rmse_val = rmse(pinn_pred, ds["u_test_true"])
        pinn_rmses.append(pinn_rmse_val)

        print(f"NN RMSE   : {nn_rmse_val:.6f}")
        print(f"PINN RMSE : {pinn_rmse_val:.6f}")

    # Plot
    ensure_dir("figs")
    plt.figure()
    plt.plot(train_sizes, nn_rmses, marker="o", label="NN (data-driven)")
    plt.plot(train_sizes, pinn_rmses, marker="o", label="PINN")
    plt.xlabel("Number of training samples")
    plt.ylabel("Test RMSE")
    plt.title("Effect of training data volume on NN vs PINN")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/data_volume_study.png", dpi=200)
    plt.close()

    print("\n=== Data volume study summary ===")
    for n, e_nn, e_pinn in zip(train_sizes, nn_rmses, pinn_rmses):
        print(f"n_train={n:3d} | NN RMSE={e_nn:.6f} | PINN RMSE={e_pinn:.6f}")


def main() -> None:
    # Main run for plots: lambda_phys = 1.0 (commonly a reasonable start)
    _, metrics_main = run_once(lambda_phys=1.0)

    print("\n=== Main metrics (dense test grid vs analytic true) ===")
    for k, v in metrics_main.items():
        print(f"{k:10s}: {v:.6f}")

    # Sensitivity sweep over physics weight
    lambdas = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]
    rmses = []
    all_metrics = {}

    for lam in lambdas:
        print(f"\n--- Sensitivity run: lambda_phys={lam} ---")
        pinn_rmse, metrics = run_once(lambda_phys=lam if lam > 0 else 1e-12)  # avoid zero in PINN loop
        rmses.append(pinn_rmse)
        all_metrics[lam] = metrics

    # Plot sensitivity
    ensure_dir("figs")
    plot_sensitivity("figs/sensitivity_lambda.png", [max(l, 1e-12) for l in lambdas], rmses)

    print("\n=== Sensitivity summary (PINN test RMSE) ===")
    for lam, e in zip(lambdas, rmses):
        print(f"lambda_phys={lam:>7g}  ->  RMSE={e:.6f}")

    print("\nFigures saved to ./figs/ (nn_fit.png, pinn_fit.png, training_curves.png, sensitivity_lambda.png)")


# 9. NN architecture comparison (EXAM REQUIREMENT)

def run_nn_architecture_study() -> None:
    """
    Compare different NN architectures for the purely data-driven model.
    This directly fulfills the exam requirement to test different NN settings.
    """
    set_seed(42)

    data_cfg = DataConfig()
    ds = make_dataset(data_cfg)

    architectures = {
        "2x32": [1, 32, 32, 1],
        "3x64": [1, 64, 64, 64, 1],
        "4x64": [1, 64, 64, 64, 64, 1],
        "3x128": [1, 128, 128, 128, 1],
    }

    rmses = {}

    for name, layers in architectures.items():
        print(f"\n--- NN architecture: {name} ---")

        model = MLP(layers, activation=nn.Tanh())
        cfg = TrainConfig(
            lambda_phys=0.0,
            epochs=5000,
            patience=350,
            lr=1e-3
        )

        model, _ = train_conventional_nn(
            model,
            ds["t_train"], ds["u_train"],
            ds["t_val"], ds["u_val"],
            cfg
        )

        pred = evaluate_model_on_grid(model, ds["t_test"], cfg.device)
        err = rmse(pred, ds["u_test_true"])
        rmses[name] = err

        print(f"Test RMSE ({name}) = {err:.6f}")

    # Plot
    ensure_dir("figs")
    plt.figure()
    plt.bar(rmses.keys(), rmses.values())
    plt.ylabel("Test RMSE")
    plt.xlabel("NN architecture")
    plt.title("Effect of NN architecture on extrapolation performance")
    plt.tight_layout()
    plt.savefig("figs/nn_architecture_study.png", dpi=200)
    plt.close()

    print("\n=== NN Architecture Study Summary ===")
    for k, v in rmses.items():
        print(f"{k:>6s} : RMSE = {v:.6f}")


if __name__ == "__main__":
    main()

    run_data_volume_study(
        train_sizes=[10, 25, 50, 100],
        lambda_phys=1.0
    )

    run_nn_architecture_study()
