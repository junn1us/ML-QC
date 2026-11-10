"""Render report figures/tables from saved outputs, without refitting any chart.

Run with the starter environment. Default paths are relative to this script,
not the shell working directory. Generated artifacts can be regenerated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = ["XbarS", "Hotelling", "PCA", "OneClassSVM", "CUSUM", "EWMA"]
NAMES = ["X-bar / S", "Directional T²", "PCA T² / Q", "One-class SVM", "Loss CUSUM", "Loss EWMA"]
TEX_NAMES = [r"$\bar{x}/S$", r"Directional $T^2$", r"PCA $T^2/Q$", "One-class SVM", "Loss CUSUM", "Loss EWMA"]
COLORS = ["#126E82", "#A33B20", "#6E59A5", "#AD6C00", "#21805C", "#C64D7E"]
BLUE, RED, PURPLE = "#126E82", "#B93E36", "#7553A0"


def tex_table(headers, rows, align):
    body = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(str(v) for v in row) + r" \\" for row in rows)
    return "\n".join(body + [r"\bottomrule", r"\end{tabular}", ""])


def render(source, destination):
    meta = json.loads((source / "run_metadata.json").read_text())
    if meta["mode"] != "synthetic_toy":
        raise ValueError("This report template describes the synthetic demonstration only")
    sheet = pd.read_csv(source / "sheet_trace.csv")
    groups = pd.read_csv(source / "subgroup_trace.csv")
    ct = pd.read_csv(source / "cusum_table.csv")
    et = pd.read_csv(source / "ewma_table.csv")
    cal = pd.read_csv(source / "calibration.csv").set_index("method").loc[METHODS]
    val = pd.read_csv(source / "validation.csv").set_index("method").loc[METHODS]
    faults = pd.read_csv(source / "fault_comparison.csv")
    models, limits = meta["models"], meta["models"]["score_limits"]
    change = meta["trace"]["first_changed_sheet"]
    figures, tables = destination / "figures", destination / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True,
                         "grid.alpha": 0.18, "pdf.fonttype": 42})

    def save(fig, name):
        for suffix in ("pdf", "png"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def panel(ax, x, y, title, ylabel, bounds, center=None, first=None):
        x, y = np.asarray(x), np.asarray(y)
        ax.axvspan(change, 500, color=PURPLE, alpha=.055)
        ax.plot(x, y, color=BLUE, lw=.9)
        alarm = np.zeros(len(y), dtype=bool)
        for side, bound in bounds:
            bound = np.broadcast_to(np.asarray(bound), y.shape)
            ax.plot(x, bound, color=RED, linestyle="--", lw=1.2)
            alarm |= y > bound if side == "upper" else y < bound
        ax.scatter(x[alarm], y[alarm], color=RED, s=13, zorder=4)
        if center is not None:
            ax.axhline(center, color="#65747C", lw=.8, linestyle=":")
        ax.axvline(change, color=PURPLE, linestyle="--", lw=1.3)
        if first is not None:
            ax.axvline(first, color="#AD6C00", linestyle=":", lw=1.3)
        ax.set(title=title, ylabel=ylabel, xlabel="Sheet (group-end sheet for X-bar / S)", xlim=(1, 500))

    def finish(fig, title):
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.legend(handles=[Line2D([], [], color=BLUE, label="Statistic"),
                            Line2D([], [], color=RED, ls="--", label="Calibrated limit"),
                            Line2D([], [], marker="o", color=RED, ls="", label="Component exceedance"),
                            Line2D([], [], color=PURPLE, ls="--", label=f"Change: sheet {change}"),
                            Line2D([], [], color="#AD6C00", ls=":", label="First rule alarm")],
                   loc="outside lower center", ncol=3, frameon=False, fontsize=9)

    first = meta["trace"]["first_alarm_sheet"]
    ml = models["XbarS"]["limits_mm"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.7), layout="constrained")
    panel(axes[0], groups.sheet_end, groups.xbar_mm, "Subgroup mean of maximum |Uz|", "Mean (mm)",
          [("lower", ml["xbar_lcl"]), ("upper", ml["xbar_ucl"])], ml["xbar_cl"], first["XbarS"])
    panel(axes[1], groups.sheet_end, groups.s_mm, "Within-subgroup sample standard deviation", "SD (mm)",
          [("lower", ml["s_lcl"]), ("upper", ml["s_ucl"])], ml["s_cl"], first["XbarS"])
    finish(fig, "Paired X-bar / S — one calibrated OR rule")
    save(fig, "xbar_s")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout="constrained")
    specs = [("Hotelling", "hotelling_T2", "Directional T² at fixed node 44", "T²", limits["Hotelling"]),
             ("OneClassSVM", "svm_score", "One-class SVM on whitened PCs", "− decision function", limits["OneClassSVM"]),
             ("PCA", "pca_T2", "PCA retained-score T²", "T²", models["PCA"]["T2_limit"]),
             ("PCA", "pca_Q", "PCA residual Q", "Standardized residual energy", models["PCA"]["Q_limit"])]
    for ax, (method, column, title, ylabel, limit) in zip(axes.flat, specs):
        panel(ax, sheet.sheet, sheet[column], title, ylabel, [("upper", limit)], first=first[method])
    finish(fig, "Directional and profile monitoring — different sensing information")
    save(fig, "multivariate")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), layout="constrained")
    panel(axes[0], sheet.sheet, sheet.loss, "Input: loss = clip(maximum |Uz| / 3 mm, 0, 1)", "Loss", [], models["loss"]["mu"])
    panel(axes[1], ct.sheet, ct.upper, "Upper CUSUM (k = 0.5)", "Cumulative standardized loss",
          [("upper", limits["CUSUM"])], first=first["CUSUM"])
    panel(axes[2], et.sheet, et.ewma, "Upper EWMA (λ = 0.1), startup-adjusted UCL", "Smoothed loss",
          [("upper", et.ucl)], models["loss"]["mu"], first["EWMA"])
    finish(fig, "Memory charts — state is preserved across the change")
    save(fig, "memory")

    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    y = np.arange(6)
    ax.scatter(cal.restricted_mean_sheets, y - .12, color="#65747C", marker="s", label="Calibration restricted mean", zorder=3)
    ax.errorbar(val.restricted_mean_sheets, y + .12, xerr=1.96 * val.restricted_mean_mcse,
                fmt="o", color=BLUE, capsize=4, label="Validation mean ± 1.96 conditional MCSE")
    ax.axvline(meta["calibration"]["target_sheets"], color=RED, ls="--", label="Target: 370 sheets")
    ax.set(yticks=y, yticklabels=NAMES, xlabel="First-alarm run length (sheets)",
           title="Independent validation does not establish matched in-control ARLs")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "arl_validation")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.9), sharey=True, layout="constrained")
    families = [("broad_Uz_mean", "Broad Uz mean shift"), ("Ux_direction", "Ux direction shift"),
                ("local_Uz_distortion", "Localized Uz distortion")]
    for ax, (family, title) in zip(axes, families):
        for method, name, color in zip(METHODS, NAMES, COLORS):
            frame = faults[(faults.fault_family == family) & (faults.method == method)].sort_values("toy_coefficient")
            ax.plot(frame.toy_coefficient, frame.restricted_delay_sheets, "o-", ms=4, color=color, label=name)
        ax.set(title=title, xlabel="Toy coefficient (not standardized)", xticks=[.5, 1, 2], yscale="log")
    axes[0].set_ylabel("Zero-state mean delay (sheets, log scale)")
    fig.suptitle("Descriptive fault results only — unequal validated false-alarm performance", fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "fault_delays")

    with np.load(source / "example_displacements.npz", allow_pickle=False) as archive:
        U = archive["U"]
        coords = archive["measurement_coordinates"]
    if U.shape != (500, 100, 3) or coords.shape != (100, 2):
        raise ValueError("Spatial panel requires the recorded 10x10 synthetic grid")
    pre, post = U[:change - 1, :, 2].mean(axis=0), U[change - 1:, :, 2].mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    vmin, vmax = min(pre.min(), post.min()), max(pre.max(), post.max())
    for ax, values, title in zip(axes[:2], (pre, post), ("Sheets 1–200: mean Uz", "Sheets 201–500: mean Uz")):
        image = ax.imshow(values.reshape(10, 10), origin="lower", extent=(-1, 1, -1, 1), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="Normalized x", ylabel="Normalized y")
        ax.grid(False)
    fig.colorbar(image, ax=list(axes[:2]), shrink=.8, label="Mean Uz (mm)")
    delta = post - pre
    image = axes[2].imshow(delta.reshape(10, 10), origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r",
                           vmin=-np.max(np.abs(delta)), vmax=np.max(np.abs(delta)))
    axes[2].set(title="Observed mean difference", xlabel="Normalized x")
    axes[2].grid(False)
    fig.colorbar(image, ax=axes[2], shrink=.8, label="Difference (mm)")
    fig.suptitle("Toy measurement grid — not candidate fixture positions", fontweight="bold")
    save(fig, "spatial")

    rows = [[name, f"{cal.loc[m, 'restricted_mean_sheets']:.2f}", f"{val.loc[m, 'restricted_mean_sheets']:.2f}",
             f"{val.loc[m, 'restricted_mean_mcse']:.2f}", f"{100*val.loc[m, 'censored_fraction']:.1f}\\%"]
            for name, m in zip(TEX_NAMES, METHODS)]
    (tables / "validation.tex").write_text(tex_table(["Method", "Calibration", "Validation", "MCSE", "Censored"], rows, "lrrrr"))
    rows = [[name, f"{limits[m]:.5f}", str(int(val.loc[m, "unit_sheets"]))] for name, m in zip(TEX_NAMES, METHODS)]
    (tables / "thresholds.tex").write_text(tex_table(["Rule", "Score threshold", "Sheets/update"], rows, "lrr"))
    rows = []
    for name, method in zip(TEX_NAMES, METHODS):
        event = first[method]
        status = "No alarm by 500" if event is None else ("Pre-change alarm" if event < change else "Post-change alarm")
        delay = "---" if event is None or event < change else str(event - change + 1)
        rows.append([name, "---" if event is None else str(event), status, delay])
    (tables / "first_alarms.tex").write_text(tex_table(["Rule", "First alarm", "Interpretation", "Post-change delay"], rows, "lrll"))
    rows = []
    for family, title in families:
        for coeff in [.5, 1, 2]:
            f = faults[(faults.fault_family == family) & (faults.toy_coefficient == coeff)].set_index("method")
            rows.append([title, f"{coeff:g}"] + [f"{f.loc[m, 'restricted_delay_sheets']:.1f}" for m in METHODS])
    (tables / "faults.tex").write_text(tex_table(["Family", "$c$", r"$\bar{x}/S$", r"$T^2$", "PCA", "SVM", "CUSUM", "EWMA"], rows, "llrrrrrr"))
    manifest = {"input_directory": str(source.resolve()), "mode": meta["mode"],
                "refit_performed": False, "trace_first_alarm_sheet": first,
                "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source.iterdir() if p.suffix in (".csv", ".json", ".npz")},
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figure_files": sorted(p.name for p in figures.iterdir()),
                "interval_convention": "Validation mean +/- 1.96 conditional MCSE; approximate, excludes fit/pool uncertainty"}
    (destination / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered six figure pairs and four LaTeX tables in {destination}")
    print("First alarms:", first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "demo_output")
    parser.add_argument("--out", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    render(args.source, args.out)