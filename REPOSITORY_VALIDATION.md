# Standalone repository validation

Packaging date: November 10, 2026.

## Scope and preservation

This repository is a standalone copy of the completed QC starter, not the parent
RL/fixture-training workspace. All nine root Python source/test files were copied
byte-for-byte and checked against the originals. Frozen study inputs and results
were retained without rerunning the large experiment or tuning any limits.

Original files remain untouched. The copied manuscript needed exactly one syntax
repair: the invalid hyperref option `allcawolors` was corrected to `allcolors`.
Other manuscript content was preserved. The PDF was rebuilt from the corrected
copy, and its verification record refreshed rather than reusing a stale PDF audit.

- Original, user-edited manuscript SHA-256 at packaging:
  `5443faaf384917831195688236ee00da8c2148d6e215b768795b90e06ae61dba`
- Packaged manuscript SHA-256:
  `32f649e43a87f06cba68716667e83012a850e5bac1d21472378a6d4981cbf940`

Documentation was adapted to standalone paths. The detailed original README is
retained as [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md), with portable setup
instructions. Historical environment observations remain in [VALIDATION.md](VALIDATION.md).

## Executed checks

- **40/40 unittest tests passed** from the copied repository root.
- Core modules imported from the copied repository, not the original source folder.
- Both reports compiled using Tectonic 0.17.0 without reported build warnings.
- The exploratory report has **8 pages**.
- The latest manuscript has **10 pages**: four main, one references, five appendix.
- The full existing report verifier passed without weakening its checks:
  source/plan hashes, distinct stream identifiers, batch totals, exact binomial
  interval cross-checks, geometric transforms, simultaneous matching criterion,
  authentic publisher class, numerical PDF content, appendix figures, clean
  LaTeX warnings and page structure. See [the audit record](iise_report/verification.json).

Validation reused the existing Python 3.14.2 environment and external compiler;
neither is bundled. This is not a clean-machine dependency-installation test.
Direct dependencies are pinned; transitive dependencies and OS libraries are not
fully locked. No full 5,205,000-sheet rerun was performed during packaging.
Programmatic checks do not replace visual proofreading or physical validation.

## Distribution boundaries

Included: source/tests, direct dependency declarations, both report sources and
PDFs, figures/tables, toy sample outputs, the frozen matched model and results,
analysis plans, provenance, and the original publisher template archive needed
for authenticity checks.

Excluded: virtual environments, caches, compiler binaries, unrelated training
checkpoints/code, duplicate report-source archives, and LaTeX build intermediates.
Build logs are generated locally for the verifier but are not tracked or packaged.

Historical JSON metadata retains original interpreter/output paths for provenance.
These are not executable dependencies and are not used to resolve input files.
They may contain the original local username; review these records before public
publication rather than silently rewriting scientific provenance.

No remote or public hosting is configured, and no initial commit is made on the
owner's behalf. The local Git index stages the intended distribution files for
review. The companion ZIP contains those files without the Git directory.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before assigning a license.