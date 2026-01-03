import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# 0. Config
SEED = 0
DEVICE = "cpu"
OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

P_MIN, P_MAX = 1.0, 4.0
N_TRAIN = 2000
N_VAL = 500
EPOCHS = 4000
LR = 1e-3

# penalty weights
LAMBDA_C = 100.0   # constraints (annulus + x>=y)
LAMBDA_B = 10.0    # box bounds penalty

BOX_MIN, BOX_MAX = -5.0, 5.0

# baseline comparison
N_COMPARE = 25
BASELINE_GRID_N = 181

# logging cadence
VAL_CHECK_EVERY = 200

torch.manual_seed(SEED)
np.random.seed(SEED)


# 0b. Flowchart
#deprecated, left for reference, but created in diagrams software, couldn't draw with Python properly
def save_flowchart(out_path: str):
    """
    Saves a simple flowchart of the differentiable learning solver pipeline.
    Produces: figs/ex3_flowchart.png
    """
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(10, 3))
    ax = plt.gca()
    ax.set_axis_off()

    boxes = [
        ("Sample $p \\sim \\mathcal{U}[1,4]$\n(train/val)", (0.02, 0.35)),
        ("NN forward\n$(x,y)=\\mathrm{NN}(p;\\theta)$", (0.25, 0.35)),
        ("Compute objective\n$f(x,y)$", (0.48, 0.55)),
        ("Compute constraints\npenalties via ReLU", (0.48, 0.15)),
        ("Loss\n$\\mathcal{L}=f+\\lambda\\,c+\\lambda_b\\,b$", (0.70, 0.35)),
        ("Backprop + Adam\nupdate $\\theta$", (0.88, 0.35)),
    ]

    box_w, box_h = 0.18, 0.26
    for text, (x, y) in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x + box_w/2, y + box_h/2, text, ha="center", va="center", fontsize=9)

    # arrows
    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=1)
        )

    # sample -> NN -> loss -> update
    arrow(0.02 + box_w, 0.35 + box_h/2, 0.25, 0.35 + box_h/2)
    arrow(0.25 + box_w, 0.35 + box_h/2, 0.70, 0.35 + box_h/2)
    arrow(0.70 + box_w, 0.35 + box_h/2, 0.88, 0.35 + box_h/2)

    # NN -> objective and NN -> constraints
    arrow(0.25 + box_w, 0.35 + box_h/2, 0.48, 0.55 + box_h/2)
    arrow(0.25 + box_w, 0.35 + box_h/2, 0.48, 0.15 + box_h/2)

    # objective -> loss and constraints -> loss
    arrow(0.48 + box_w, 0.55 + box_h/2, 0.70, 0.35 + box_h/2)
    arrow(0.48 + box_w, 0.15 + box_h/2, 0.70, 0.35 + box_h/2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# 1. Objectives
def himmelblau(x, y):
    return (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2


def mccormick(x, y):
    return torch.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1.0


# 2. NN Solver
class ParametricSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, p):
        return self.net(p)


# 3. Constraint penalties
def constraint_penalties(x, y, p):
    """
    Returns mean penalties for:
      c1: (p/2)^2 <= r^2
      c2: r^2 <= p^2
      c3: x >= y
      b : -5<=x,y<=5
    """
    r2 = x**2 + y**2

    c1 = torch.relu((p / 2.0) ** 2 - r2)
    c2 = torch.relu(r2 - p**2)
    c3 = torch.relu(y - x)

    bx = torch.relu(x - BOX_MAX) + torch.relu(BOX_MIN - x)
    by = torch.relu(y - BOX_MAX) + torch.relu(BOX_MIN - y)
    b = bx + by

    return c1.mean(), c2.mean(), c3.mean(), b.mean()


# 4. Baseline solver (grid search)
def grid_search_baseline(p_values, obj_name, grid_n=201):
    """
    Brute-force grid search on [-5,5]^2 subject to constraints.
    Returns x*, y* arrays for each p.
    """
    xs = np.linspace(BOX_MIN, BOX_MAX, grid_n)
    ys = np.linspace(BOX_MIN, BOX_MAX, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    R2 = X**2 + Y**2

    x_star = np.zeros(len(p_values), dtype=np.float64)
    y_star = np.zeros(len(p_values), dtype=np.float64)

    for i, p in enumerate(p_values):
        inner = (p / 2.0) ** 2
        outer = p**2
        feasible = (R2 >= inner) & (R2 <= outer) & (X >= Y)

        if not np.any(feasible):
            x_star[i] = np.nan
            y_star[i] = np.nan
            continue

        if obj_name == "himmelblau":
            F = (X**2 + Y - 11.0) ** 2 + (X + Y**2 - 7.0) ** 2
        elif obj_name == "mccormick":
            F = np.sin(X + Y) + (X - Y) ** 2 - 1.5 * X + 2.5 * Y + 1.0
        else:
            raise ValueError("obj_name must be 'himmelblau' or 'mccormick'")

        F_feas = np.where(feasible, F, np.inf)
        idx = np.unravel_index(np.argmin(F_feas), F_feas.shape)
        x_star[i] = X[idx]
        y_star[i] = Y[idx]

    return x_star, y_star


# 5. Training loop
def train_for_objective(obj_name: str):
    if obj_name not in ["himmelblau", "mccormick"]:
        raise ValueError("obj_name must be 'himmelblau' or 'mccormick'")

    model = ParametricSolver().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train/val p samples
    p_train = (torch.rand(N_TRAIN, 1) * (P_MAX - P_MIN) + P_MIN).to(DEVICE)
    p_val = (torch.rand(N_VAL, 1) * (P_MAX - P_MIN) + P_MIN).to(DEVICE)

    # plot train/val p distribution
    plt.figure()
    plt.hist(p_train.cpu().numpy().ravel(), bins=30, alpha=0.6, label="train")
    plt.hist(p_val.cpu().numpy().ravel(), bins=30, alpha=0.6, label="val")
    plt.xlabel("p")
    plt.ylabel("count")
    plt.title(f"Parameter samples distribution (p) - {obj_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_p_distribution_{obj_name}.png"), dpi=200)
    plt.close()

    # Histories
    hist_total, hist_obj = [], []
    hist_c1, hist_c2, hist_c3, hist_box = [], [], [], []
    val_epochs, val_feas_scores = [], []

    # Training
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        xy = model(p_train)
        x, y = xy[:, 0], xy[:, 1]
        p = p_train[:, 0]

        if obj_name == "himmelblau":
            obj = himmelblau(x, y)
        else:
            obj = mccormick(x, y)

        c1m, c2m, c3m, bm = constraint_penalties(x, y, p)

        loss = obj.mean() + LAMBDA_C * (c1m + c2m + c3m) + LAMBDA_B * bm
        loss.backward()
        optimizer.step()

        hist_total.append(float(loss.item()))
        hist_obj.append(float(obj.mean().item()))
        hist_c1.append(float(c1m.item()))
        hist_c2.append(float(c2m.item()))
        hist_c3.append(float(c3m.item()))
        hist_box.append(float(bm.item()))

        if epoch % VAL_CHECK_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                xyv = model(p_val)
                xv, yv = xyv[:, 0], xyv[:, 1]
                pv = p_val[:, 0]
                c1v, c2v, c3v, bv = constraint_penalties(xv, yv, pv)
                feas_score = float((c1v + c2v + c3v + bv).item())
            val_epochs.append(epoch)
            val_feas_scores.append(feas_score)

    # training loss
    plt.figure()
    plt.semilogy(hist_total)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss (log)")
    plt.title(f"Training loss (log scale) - {obj_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_training_loss_{obj_name}.png"), dpi=200)
    plt.close()

    # objective vs penalties
    plt.figure()
    plt.semilogy(hist_obj, label="objective mean")
    plt.semilogy(np.array(hist_c1) * LAMBDA_C, label="inner penalty * λ")
    plt.semilogy(np.array(hist_c2) * LAMBDA_C, label="outer penalty * λ")
    plt.semilogy(np.array(hist_c3) * LAMBDA_C, label="x>=y penalty * λ")
    plt.semilogy(np.array(hist_box) * LAMBDA_B, label="box penalty * λb")
    plt.xlabel("Epoch")
    plt.ylabel("Value (log)")
    plt.title(f"Objective vs constraint penalties - {obj_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_obj_vs_penalties_{obj_name}.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.semilogy(val_epochs, val_feas_scores, marker="o", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Mean feasibility proxy (log)")
    plt.title(f"Validation constraint-violation proxy - {obj_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_val_feasibility_{obj_name}.png"), dpi=200)
    plt.close()

    model.eval()
    with torch.no_grad():
        xyv = model(p_val)
        xv = xyv[:, 0].cpu().numpy()
        yv = xyv[:, 1].cpu().numpy()
        pv = p_val[:, 0].cpu().numpy()

    plt.figure()
    plt.scatter(xv, yv, s=8, alpha=0.35)
    plt.xlabel("x (NN)")
    plt.ylabel("y (NN)")
    plt.title(f"Predicted solutions on validation set - {obj_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_solution_scatter_{obj_name}.png"), dpi=200)
    plt.close()

    # r^2 vs bounds (checks annulus constraint directly)
    r2 = xv**2 + yv**2
    lower = (pv / 2.0) ** 2
    upper = pv**2
    plt.figure()
    plt.scatter(pv, r2, s=8, alpha=0.35, label="$r^2=x^2+y^2$")
    plt.scatter(pv, lower, s=8, alpha=0.35, label="$(p/2)^2$")
    plt.scatter(pv, upper, s=8, alpha=0.35, label="$p^2$")
    plt.xlabel("p")
    plt.ylabel("value")
    plt.title(f"Annulus constraint check on validation set - {obj_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_annulus_check_{obj_name}.png"), dpi=200)
    plt.close()

    # Baseline comparison
    p_compare = np.linspace(P_MIN, P_MAX, N_COMPARE)
    t0 = time.time()
    x_base, y_base = grid_search_baseline(p_compare, obj_name=obj_name, grid_n=BASELINE_GRID_N)
    t_baseline = time.time() - t0

    p_t = torch.tensor(p_compare.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        xy_hat = model(p_t).cpu().numpy()
    x_hat, y_hat = xy_hat[:, 0], xy_hat[:, 1]

    def obj_np(x, y):
        if obj_name == "himmelblau":
            return (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2
        return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1.0

    f_base = obj_np(x_base, y_base)
    f_hat = obj_np(x_hat, y_hat)

    plt.figure()
    plt.plot(p_compare, f_base, label="baseline grid search")
    plt.plot(p_compare, f_hat, label="NN prediction")
    plt.xlabel("p")
    plt.ylabel("Objective value f(x,y)")
    plt.title(f"Objective comparison vs baseline - {obj_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ex3_obj_comparison_{obj_name}.png"), dpi=200)
    plt.close()

    # Summary for appendix/logs + helps justify the numbers in your report text
    mean_gap = float(np.nanmean(np.abs(f_hat - f_base)))
    # feasibility quick stats on val
    xvt = torch.tensor(xv, dtype=torch.float32)
    yvt = torch.tensor(yv, dtype=torch.float32)
    pvt = torch.tensor(pv, dtype=torch.float32)
    c1v, c2v, c3v, bv = constraint_penalties(xvt, yvt, pvt)

    print(f"\n=== Exercise 3 summary: {obj_name} ===")
    print(f"Saved figures to: {OUT_DIR}/")
    print(f"Baseline grid search time ({N_COMPARE} p values): {t_baseline:.2f} s")
    print(f"Objective gap (mean |f_hat - f_base|): {mean_gap:.4f}")
    print(f"Val mean penalties: c1={float(c1v.item()):.3e}, c2={float(c2v.item()):.3e}, "
          f"c3={float(c3v.item()):.3e}, box={float(bv.item()):.3e}")

    return mean_gap, model


def main():
    # required by exercise: show a flowchart of the method
    save_flowchart(os.path.join(OUT_DIR, "ex3_flowchart.png"))

    gap_h, _ = train_for_objective("himmelblau")
    gap_m, _ = train_for_objective("mccormick")

    # Keep these numbers aligned with what you state in the report text
    print("\n=== Report numbers (copy into LaTeX text) ===")
    print(f"Himmelblau mean objective gap: {gap_h:.4f}")
    print(f"McCormick mean objective gap:  {gap_m:.4f}")


if __name__ == "__main__":
    main()
