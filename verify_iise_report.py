"""Audit the report against locked numerical outputs; requires pypdf.

This verifies machine-readable content and layout diagnostics, not visual
proofreading or physical validation. It does not fit, calibrate, or tune.
"""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'iise_report'
RESULT = REPORT / 'results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = json.loads((RESULT / 'study_metadata.json').read_text())
    plan = json.loads((RESULT / 'locked_plan.json').read_text())
    frozen = json.loads((RESULT / 'frozen_before_phase2.json').read_text())
    assert (RESULT / 'locked_plan.json').read_bytes() == (REPORT / 'experiment_plan.json').read_bytes()
    assert frozen['plan_sha256'] == sha(RESULT / 'locked_plan.json')
    assert frozen == metadata['frozen_phase1']
    for name, digest in metadata['source_sha256'].items():
        assert sha(ROOT / name) == digest
    keys = [tuple(s['spawn_key']) for s in metadata['streams']]
    assert len(set(keys)) == len(keys) == 9
    val = pd.read_csv(RESULT / 'phase2_validation.csv')
    faults = pd.read_csv(RESULT / 'phase2_faults.csv')
    for frame in (val, faults):
        for r in frame.itertuples():
            ci = binomtest(int(r.alarms), int(r.sheets)).proportion_ci(confidence_level=.95, method='exact')
            np.testing.assert_allclose([r.p_lo, r.p_hi], [ci.low, ci.high], atol=1e-11)
            np.testing.assert_allclose(r.arl_hat, r.sheets/r.alarms)
            np.testing.assert_allclose([r.arl_lo, r.arl_hi], [1/r.p_hi, 1/r.p_lo])
            np.testing.assert_allclose(r.detection_by_h, 1-(1-r.p_hat)**50)
    def check_batches(frame, path):
        b = pd.read_csv(path)
        assert b.sheets.sum() == frame.sheets.iloc[0]
        by_method = frame.set_index('method')
        assert b.PCA_alarms.sum() == by_method.loc['PCA_T2_Q', 'alarms']
        assert b.Shewhart_alarms.sum() == by_method.loc['Shewhart_d', 'alarms']
        assert (b.both_alarms <= b.PCA_alarms).all()
        assert (b.both_alarms <= b.Shewhart_alarms).all()
    check_batches(val, RESULT / 'phase2_ic_batch_counts.csv')
    for scenario in plan['fault_scenarios']:
        family, coefficient = scenario['family'], scenario['coefficient']
        subset = faults[(faults.family == family) & (faults.coefficient == coefficient)]
        check_batches(subset, RESULT / f'fault_{family}_{coefficient:g}_batch_counts.csv')
    comp = metadata['phase2_comparison']
    assert comp['match_criterion_passed']
    assert .9 <= comp['simultaneous_ratio_lo'] <= comp['simultaneous_ratio_hi'] <= 1.1
    for method, row in comp['simultaneous_marginal_intervals'].items():
        assert 333 <= row['arl_lo'] <= row['arl_hi'] <= 407
        ci = binomtest(row['alarms'], row['sheets']).proportion_ci(confidence_level=.975, method='exact')
        np.testing.assert_allclose([row['p_lo'], row['p_hi']], [ci.low, ci.high], atol=1e-11)
    assets = json.loads((REPORT / 'assets/manifest.json').read_text())
    for name, digest in assets['inputs_sha256'].items():
        assert sha(RESULT / name) == digest
    assert sha(ROOT / 'build_iise_assets.py') == assets['renderer_sha256']
    with zipfile.ZipFile(REPORT / 'template/interactcadlatex.zip') as archive:
        assert archive.read('interact.cls') == (REPORT / 'interact.cls').read_bytes()
    pdf = PdfReader(REPORT / 'IISE_QC_PhaseI_PhaseII.pdf')
    assert len(pdf.pages) == 10, 'Expected four main-text pages + references + five appendix pages'
    pages = [p.extract_text() for p in pdf.pages]
    assert all(len(p) > 2000 for p in pages[:4])
    assert pages[4].startswith('References')
    assert 'ABSTRACT' in pages[0].upper() or 'abstract' in (REPORT / 'IISE_QC_PhaseI_PhaseII.tex').read_text()
    assert 'Problem' in pages[0] and 'Statistical validation' in pages[1]
    assert 'Results' in pages[2] and 'limitations' in pages[3]
    # PDF math extraction can insert a space between the integer and decimal.
    result_text = ''.join(pages[2].split())
    for v in ['5,335', '5,489', '374.9', '364.4', '0.968', '1.094', '1.000019']:
        assert v in result_text, v
    assert '10489779' in pages[4]
    appendix_text = ''.join(''.join(pages[5:]).split()).lower()
    for label in ['supplementarycontrol-chartvisuals', 'hotelling', 'cusum', 'ewma', 'spatial']:
        assert label in appendix_text, label
    figure_hashes = {}
    for name in ['xbar_s', 'multivariate', 'memory', 'arl_validation', 'fault_delays', 'spatial']:
        copied = REPORT / 'assets/exploratory' / f'{name}.pdf'
        assert len(PdfReader(copied).pages) == 1
        original = ROOT / 'report/figures' / f'{name}.pdf'
        if original.exists():
            assert copied.read_bytes() == original.read_bytes(), name
        figure_hashes[name] = sha(copied)
    log = (REPORT / 'IISE_QC_PhaseI_PhaseII.log').read_text()
    for bad in ['Overfull', 'Undefined control sequence', 'undefined references', 'Citation ', 'LaTeX Warning:']:
        assert bad not in log, bad
    report = {'status': 'passed', 'main_text_pages': 4, 'references_pages': 1, 'appendix_pages': 5,
              'exploratory_figure_sha256': figure_hashes,
              'pdf_sha256': sha(REPORT / 'IISE_QC_PhaseI_PhaseII.pdf'),
              'manuscript_sha256': sha(REPORT / 'IISE_QC_PhaseI_PhaseII.tex'),
              'appendix_source_sha256': sha(REPORT / 'control_chart_appendix.tex'),
              'verification': ['locked plan and code hashes', 'independent seed stream identifiers',
                               'all alarm counts agree with batch records', 'independent scipy binomtest interval cross-check',
                               'geometric transformations', 'prespecified simultaneous matching criterion',
                               'unchanged authentic publisher class', 'four main-text pages, references fifth',
                               'all six exploratory figure files and appendix labels',
                               'numerical values in PDF', 'clean LaTeX warnings check'],
              'limitation': 'Programmatic verification, not physical validation or a visual page inspection.'}
    (REPORT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()