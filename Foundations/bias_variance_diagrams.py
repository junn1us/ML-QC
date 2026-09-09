"""Visualize the fixed-x0 bias–variance proof in six annotated diagrams.

Requires numpy and matplotlib. Run this file to save PNG/PDF copies and open
the figure. Optional arguments: --no-show, --output-dir, --f0, --bias,
--noise-std, --model-std, and --theme light|dark. Dark exports use a _dark
suffix to preserve the light copies. No external LaTeX installation is needed.

The normal distributions are illustrative, not assumptions of the proof.
Only finite second moments and independent, mean-zero test noise are needed.
All expectations over fitted predictions refer to repeated training samples.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np


THEMES = {
    "light": {
        "background": "#edf2f8", "surface": "#ffffff",
        "ink": "#17263c", "muted": "#53657d", "edge": "#c8d2e0",
        "noise": "#168b93", "bias": "#d17a20", "model": "#7454be",
        "truth": "#304d72", "bar_text": "#ffffff",
    },
    "dark": {
        "background": "#10151f", "surface": "#1b2433",
        "ink": "#edf3ff", "muted": "#b3c1d6", "edge": "#4b5c75",
        "noise": "#54d6ce", "bias": "#ffbb66", "model": "#b99aff",
        "truth": "#8abaff", "bar_text": "#10151f",
    },
}


def normal_pdf(x, mean, std):
    """Normal density used only to illustrate the moment identities."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


@matplotlib.rc_context()
def build_figure(f0=2.0, bias=1.2, noise_std=0.8, model_std=0.65, theme="light"):
    """Return a light/dark proof figure without changing global plot styles."""
    import matplotlib.pyplot as plt

    if theme not in THEMES:
        raise ValueError("Theme must be 'light' or 'dark'.")
    palette = THEMES[theme]
    BG, INK, MUTED = (palette[key] for key in ("background", "ink", "muted"))
    NOISE, BIAS, MODEL, TRUTH = (palette[key] for key in ("noise", "bias", "model", "truth"))
    if not all(np.isfinite(v) for v in (f0, bias, noise_std, model_std)):
        raise ValueError("All parameters must be finite.")
    if noise_std <= 0 or model_std <= 0:
        raise ValueError("Standard deviations must be positive.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.facecolor": palette["surface"], "axes.edgecolor": palette["edge"],
        "legend.labelcolor": INK,
        "mathtext.fontset": "dejavusans", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    mean_prediction = f0 + bias
    noise_var, model_var = noise_std**2, model_std**2
    total = noise_var + bias**2 + model_var
    spread = max(noise_std, model_std)
    x = np.linspace(min(f0, mean_prediction) - 4 * spread,
                    max(f0, mean_prediction) + 4 * spread, 1000)

    fig = plt.figure(figsize=(18, 16), facecolor=BG)
    fig.text(0.045, 0.962, "WHERE PREDICTION ERROR COMES FROM",
             fontsize=25, weight="bold")
    fig.text(0.045, 0.936,
             r"Bias–variance decomposition at a fixed input $x_0$  |  "
             r"$f_0=f(x_0)$,  $\hat f_0=\hat f_{\mathcal{D}}(x_0)$,  "
             r"$m=E_{\mathcal{D}}[\hat f_0]$", color=MUTED, fontsize=13)
    grid = fig.add_gridspec(3, 2, left=0.04, right=0.96, bottom=0.09,
                           top=0.91, hspace=0.10, wspace=0.055)

    def card(row, col, number, title, subtitle):
        ax = fig.add_subplot(grid[row, col])
        ax.set_facecolor(palette["surface"])
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.035, 0.91, f"{number:02d}", color=MODEL, weight="bold", fontsize=14)
        ax.text(0.105, 0.91, title, fontsize=15, weight="bold")
        ax.text(0.035, 0.82, subtitle, color=MUTED, fontsize=10.5)
        return ax

    def equation(ax, text, y=0.075, size=13):
        ax.text(0.5, y, text, ha="center", va="center", fontsize=size,
                bbox={"boxstyle": "round,pad=0.5", "fc": BG, "ec": "none"})

    def plot_area(ax):
        child = ax.inset_axes([0.10, 0.27, 0.85, 0.46])
        child.tick_params(labelsize=9)
        child.spines["left"].set_color(palette["edge"])
        child.spines["bottom"].set_color(palette["edge"])
        return child

    # 1. The two independent sources of randomness.
    ax = card(0, 0, 1, "Separate the randomness",
              "The truth is fixed. Training samples and fresh test noise are random.")
    for y, left, right, color in [
        (0.64, r"Training sample $\mathcal{D}$", r"Fitted prediction $\hat f_0$", MODEL),
        (0.38, r"New noise $\varepsilon$", r"Test response $Y=f_0+\varepsilon$", NOISE),
    ]:
        for xpos, text in [(0.25, left), (0.75, right)]:
            ax.text(xpos, y, text, ha="center", va="center", fontsize=12,
                    bbox={"boxstyle": "round,pad=0.75", "fc": palette["surface"], "ec": color, "lw": 1.6})
        ax.annotate("", xy=(0.56, y), xytext=(0.44, y),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
    ax.text(0.5, 0.505, "INDEPENDENT SOURCES", ha="center", fontsize=9, color=MUTED)
    equation(ax, r"$E[\varepsilon]=0,\quad \mathrm{Var}(\varepsilon)=\sigma^2,"
             r"\quad \varepsilon\perp\!\!\!\perp\mathcal{D}$")

    # 2. Once D is fixed, only the fresh response varies.
    ax = card(0, 1, 2, "Freeze one training sample",
              r"Condition on $\mathcal{D}$: the fitted value $a=\hat f_{\mathcal{D},0}$ is now constant.")
    p = plot_area(ax)
    a = mean_prediction + 0.8 * model_std
    conditional_x = np.linspace(min(x[0], a - spread), max(x[-1], a + spread), 1000)
    density = normal_pdf(conditional_x, f0, noise_std)
    p.fill_between(conditional_x, density, color=NOISE, alpha=0.20)
    p.plot(conditional_x, density, color=NOISE, lw=2, label=r"$Y\mid\mathcal{D}$ (illustration)")
    p.axvline(f0, color=TRUTH, ls="--", label=r"$E[Y\mid\mathcal{D}]=f_0$")
    p.axvline(a, color=MODEL, lw=2, label=r"Fixed prediction $a$")
    p.set(xlabel="Response / prediction", ylabel="Density", ylim=(0, max(density) * 1.6))
    p.legend(fontsize=8, loc="upper right", frameon=False)
    equation(ax, r"$E[(Y-a)^2\mid\mathcal{D}]=\sigma^2+(f_0-a)^2$")

    # 3. Average the conditional error over every possible fitted value.
    ax = card(1, 0, 3, "Unfreeze the training sample",
              r"Average the conditional error over $\mathcal{D}$ (the law of total expectation).")
    p = plot_area(ax)
    prediction_grid = np.linspace(min(f0, mean_prediction) - 2 * model_std,
                                  max(f0, mean_prediction) + 2 * model_std, 300)
    conditional_error = noise_var + (prediction_grid - f0)**2
    p.plot(prediction_grid, conditional_error, color=MODEL, lw=2,
           label=r"$\sigma^2+(f_0-a)^2$")
    example_predictions = mean_prediction + model_std * np.array([-1.6, -0.8, 0, 0.8, 1.6])
    p.scatter(example_predictions, noise_var + (example_predictions - f0)**2,
              color=MODEL, s=28, zorder=4, label="Example fitted values")
    p.axhline(noise_var, color=NOISE, ls="--", label=r"Noise floor $\sigma^2$")
    p.set(xlabel=r"Possible fitted prediction $a$", ylabel="Conditional MSE")
    p.legend(fontsize=8, frameon=False, loc="upper center")
    equation(ax, r"$E_{\mathcal{D},\varepsilon}[(Y-\hat f_0)^2]"
             r"=\sigma^2+E_{\mathcal{D}}[(f_0-\hat f_0)^2]$", size=12)

    # 4. Location and spread of the fitted prediction distribution.
    ax = card(1, 1, 4, "Split model error: offset + spread",
              "Bias is the mean prediction's offset; variance is spread around that mean.")
    p = plot_area(ax)
    density = normal_pdf(x, mean_prediction, model_std)
    p.fill_between(x, density, color=MODEL, alpha=0.20)
    p.plot(x, density, color=MODEL, lw=2)
    p.axvline(f0, color=TRUTH, ls="--")
    p.axvline(mean_prediction, color=MODEL, lw=2)
    peak = max(density)
    p.annotate("", xy=(mean_prediction, 1.18 * peak), xytext=(f0, 1.18 * peak),
               arrowprops={"arrowstyle": "<->", "color": BIAS, "lw": 2})
    p.text((f0 + mean_prediction) / 2, peak * 1.29, f"Bias = {bias:g}",
           ha="center", color=BIAS, fontsize=10)
    p.annotate("", xy=(mean_prediction - model_std, 0.35 * peak),
               xytext=(mean_prediction + model_std, 0.35 * peak),
               arrowprops={"arrowstyle": "<->", "color": MODEL, "lw": 2})
    p.text(mean_prediction, 0.12 * peak, r"$2\,\mathrm{SD}(\hat f_0)$",
           ha="center", fontsize=9, color=MODEL)
    p.set(xlabel=r"Fitted prediction across training samples (dashed: $f_0$; solid: $m$)",
          ylabel="Density", ylim=(0, 1.6 * peak))
    equation(ax, r"$E_{\mathcal{D}}[(\hat f_0-f_0)^2]"
             r"=\mathrm{Var}_{\mathcal{D}}(\hat f_0)+(m-f_0)^2$", size=12)

    # 5. Expand the same second-moment identity used in the proof.
    ax = card(2, 0, 5, "Why the cross term disappears",
              "The identity used twice in the proof is just a centered-square expansion.")
    ax.text(0.5, 0.66, r"$\hat f_0-f_0=(\hat f_0-m)+(m-f_0)$",
            ha="center", fontsize=17)
    ax.text(0.5, 0.55, "prediction error = fluctuation + bias",
            ha="center", fontsize=11, color=MUTED)
    ax.text(0.5, 0.42, r"$E[(\hat f_0-f_0)^2]=E[(\hat f_0-m)^2]+(m-f_0)^2$",
            ha="center", fontsize=13)
    ax.text(0.5, 0.28, r"Cross term: $2(m-f_0)\,E[\hat f_0-m]=0$",
            ha="center", fontsize=13, color=MUTED)
    ax.text(0.5, 0.19, r"because $E[\hat f_0]=m$", ha="center", fontsize=11, color=MUTED)
    equation(ax, r"$E[(Z-c)^2]=\mathrm{Var}(Z)+(E[Z]-c)^2$", size=13)

    # 6. Combine the terms into total expected test error.
    ax = card(2, 1, 6, "Add the three sources of error",
              "Exact moment values for this illustrative example (not a Monte Carlo estimate).")
    p = ax.inset_axes([0.07, 0.43, 0.86, 0.28])
    p.set_axis_off()
    left = 0.0
    components = [("Noise", noise_var, NOISE), ("Bias²", bias**2, BIAS),
                  ("Variance", model_var, MODEL)]
    for label, value, color in components:
        p.barh(0, value, left=left, height=0.6, color=color)
        if value / total > 0.09:
            p.text(left + value / 2, 0, f"{value:.3f}", ha="center", va="center",
                     color=palette["bar_text"], weight="bold", fontsize=12)
        left += value
    p.set(xlim=(0, total), ylim=(-0.5, 0.5))
    for xpos, (label, value, color) in zip([0.19, 0.5, 0.81], components):
        ax.text(xpos, 0.36, f"{label} = {value:.3f}", ha="center", color=color, fontsize=11)
    ax.text(0.5, 0.24, f"Total expected test MSE = {total:.4f}", ha="center",
            weight="bold", fontsize=14)
    equation(ax, r"$E[(Y-\hat f(x_0))^2]=\sigma^2+"
             r"\mathrm{Bias}(\hat f(x_0))^2+\mathrm{Var}(\hat f(x_0))$", size=12)

    fig.text(0.045, 0.052,
             "READING GUIDE   01 → 02 → 03: remove test noise by conditioning.   "
             "04 → 05 → 06: split the remaining model error.", fontsize=11, weight="bold")
    fig.text(0.045, 0.032,
             "Assumptions: fixed x₀, finite second moments, mean-zero fresh noise independent of training data. "
             "Gaussian curves illustrate the proof; normality is not required.",
             fontsize=10, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0", type=float, default=2.0, help="True response at x0")
    parser.add_argument("--bias", type=float, default=1.2, help="Mean prediction minus truth")
    parser.add_argument("--noise-std", type=float, default=0.8, help="Fresh-noise standard deviation")
    parser.add_argument("--model-std", type=float, default=0.65, help="Standard deviation of fitted predictions")
    parser.add_argument("--theme", choices=tuple(THEMES), default="light",
                        help="Color theme (default: light); dark exports have a _dark suffix")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "bias_variance_output")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig = build_figure(args.f0, args.bias, args.noise_std, args.model_std, theme=args.theme)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dark" if args.theme == "dark" else ""
    for extension in ("png", "pdf"):
        destination = args.output_dir / f"bias_variance_proof{suffix}.{extension}"
        fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
        print(f"Saved: {destination}")
    print(f"MSE = noise + bias² + variance = {args.noise_std**2:.4f} + "
          f"{args.bias**2:.4f} + {args.model_std**2:.4f} = "
          f"{args.noise_std**2 + args.bias**2 + args.model_std**2:.4f}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()