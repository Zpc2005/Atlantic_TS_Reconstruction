# Package integrity review

## Scope

Read-only verification of the frozen reproducibility package. No renderer, experiment, training, inference, metric recalculation, manuscript edit, or Git write was performed.

## Manifest

`SHA256_MANIFEST.csv` contains 97 unique records. All 97 listed files exist; all 97 recomputed SHA-256 values match; no duplicate path and no missing file were found.

## Scientific assets

PIRATA and WOA23 source records, exact-depth observations, coverage, M1/M2/M3 freeze/protocol records, masks, stored predictions, metrics, Table 1/4 source records, Figure 7 evidence manifest, Figure S2 selected time-series data, and bootstrap supporting records are present or explicitly traced in the package. Split and normalization definitions are retained in freeze/protocol/code records; a separate canonical standalone normalization-parameter table is not asserted where none was staged.

## Code

Selected data-preparation, model definitions, masking, metrics/evaluation, normalization, M2/M3 experiment utilities, and plotting/provenance code are staged under `03_code/`. This is a source package and does not claim that experiments can be rerun without the researcher’s environment and any separately controlled checkpoints.

## Figure provenance

11/11 current figures are represented in `04_figure_provenance/`. Each record separates scientific source from presentation/refinement lineage. Exact, near-exact, partial, and unresolved classifications are preserved. Figures 3, 5, 7, 9, and S2 remain visually unresolved; no exact visual reproduction is claimed for them.

## AI and availability facts

`AI_TOOL_USE_FACTS.md` preserves scientific traceability 11/11, AI-4 = 0, and AI-U where visual lineage is unresolved. Availability drafts contain no invented DOI or false public-repository claim and do not claim raw-archive redistribution.

## Gate result

**PASS WITH DOCUMENTED PRE-SUBMISSION GAPS**
