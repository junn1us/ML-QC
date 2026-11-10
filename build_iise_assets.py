"""Read frozen Phase II results and create manuscript tables and vector figures."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / 'iise_report'
METHODS = ['PCA_T2_Q', 'Shewhart_d']
LABELS = ['PCA T²/Q', 'Shewhart maximum']
TEX = [r'PCA $T^2/Q$', 'Shewhart maximum']


def table(headers, rows, align):
    return '\n'.join([r'\begin{tabular}{' + align + '}', r'\toprule',
                      ' & '.join(headers) + r' \\', r'\midrule'] +
                     [' & '.join(row) + r' \\' for row in rows] +
                     [r'\bottomrule', r'\end{tabular}', ''])


def main():
    result = ROOT / 'results'
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    val = pd.read_csv(result / 'phase2_validation.csv').set_index('method').loc[METHODS]
    faults = pd.read_csv(result / 'phase2_faults.csv')
    meta = json.loads((result / 'study_metadata.json').read_text())
    rows = []
    for method, name in zip(METHODS, TEX):
        r = val.loc[method]
        rows.append([name, f"{r.alarms:,.0f}", f"{1000*r.p_hat:.4f}",
                     f"{r.arl_hat:.1f} [{r.arl_lo:.1f}, {r.arl_hi:.1f}]"])
    (assets / 'validation_table.tex').write_text(table(
        ['Chart', 'Alarms / 2,000,000', r'$10^3\widehat p_0$', r'$\widehat{\mathrm{ARL}}_0$ [95\% CI]'], rows, 'lrrl'))
    rows = []
    for family, label in [('broad_Uz_mean', 'Broad'), ('local_Uz_distortion', 'Local')]:
        subset = faults[faults.family == family]
        for amplitude in sorted(subset.amplitude_parameter_mm.unique()):
            pair = subset[subset.amplitude_parameter_mm == amplitude].set_index('method')
            def interval(method):
                r = pair.loc[method]
                if r.alarms == r.sheets:
                    return r'1.000 [1.000, 1.000019]'
                return f'{r.arl_hat:.1f} [{r.arl_lo:.1f}, {r.arl_hi:.1f}]'
            rows.append([label, f'{amplitude:.3f}', interval(METHODS[0]), interval(METHODS[1]),
                         f'{pair.loc[METHODS[0], "detection_by_h"]:.3f}',
                         f'{pair.loc[METHODS[1], "detection_by_h"]:.3f}'])
    (assets / 'fault_table.tex').write_text(table(
        ['Fault', r'$a$ (mm)', r'PCA ARL$_1$ [95\% CI]', r'Shewhart ARL$_1$ [95\% CI]', r'PCA $P_{50}$', r'Shew. $P_{50}$'],
        rows, 'lrllrr'))
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), layout='constrained')
    colors = ['#176B87', '#B45A32']
    for j, (method, name, color) in enumerate(zip(METHODS, LABELS, colors)):
        r = val.loc[method]
        axes[0].errorbar(j, r.arl_hat, yerr=[[r.arl_hat-r.arl_lo], [r.arl_hi-r.arl_hat]], fmt='o', color=color, capsize=4)
    axes[0].axhline(370, color='#555555', ls='--', lw=1)
    axes[0].axhspan(333, 407, color='#aaaaaa', alpha=.12)
    axes[0].set(xticks=[0, 1], xticklabels=['PCA', 'Shewhart'], ylabel='ARL₀ (sheets)',
                title='(a) Independent in-control check', ylim=(330, 410), xlim=(-.5, 1.5))
    for ax, family, title in zip(axes[1:], ['broad_Uz_mean', 'local_Uz_distortion'], ['(b) Broad Uz shift', '(c) Local Uz distortion']):
        for method, name, color in zip(METHODS, LABELS, colors):
            f = faults[(faults.family == family) & (faults.method == method)].sort_values('amplitude_parameter_mm')
            errors = np.vstack((f.detection_by_h-f.detection_by_h_lo, f.detection_by_h_hi-f.detection_by_h))
            ax.errorbar(f.amplitude_parameter_mm, f.detection_by_h, yerr=errors, fmt='o-', color=color,
                        ms=3, capsize=2, lw=1, label=name)
        ax.axhline(val.detection_by_h.mean(), color='#888888', ls=':', lw=.8)
        ax.set(title=title, xlabel='Amplitude parameter a (mm)', ylabel='P(signal by sheet 50)', ylim=(0, 1.05))
    axes[2].legend(fontsize=6, loc='upper left', frameon=False)
    for ext in ['pdf', 'png']:
        fig.savefig(assets / f'matched_comparison.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    manifest = {'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in result.iterdir() if p.suffix in ('.csv', '.json', '.npz')},
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matched': meta['phase2_comparison']['match_criterion_passed']}
    (assets / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Wrote manuscript figure and tables:', assets)


if __name__ == '__main__':
    main()