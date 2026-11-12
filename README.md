# Retail Store Analysis

Reproducible Python analysis of the UCI Online Retail dataset (541,909 transaction lines, 1 Dec 2010 – 9 Dec 2011) for a UK-based online giftware retailer: data-quality audit, non-destructive cleaning, exploratory analysis, charts, RFM customer segmentation and revenue-recovery evidence.

No record is ever deleted or edited. Every cleaning decision is stored as a flag or derived column next to the original data, and each analysis is a filter on those flags.

## Pipeline

Run from the repository root, in this order:

| Step | Script | Output folder |
|---|---|---|
| 1. Data-quality audit (28 checks) | `python retail_audit.py data/online_retail.xlsx --out audit_output` | `audit_output/` |
| 2. Cleaning flags and derived columns | `python retail_prepare.py data/online_retail.xlsx --out prepared` | `prepared/` |
| 3. Descriptive statistics | `python retail_eda.py prepared/retail_prepared.csv.gz --out eda` | `eda/` |
| 4. Frequencies, correlation, country comparison | `python retail_eda_part3b.py prepared/retail_prepared.csv.gz --out eda/part3b` | `eda/part3b/` |
| 5. Charts | `python retail_charts.py prepared/retail_prepared.csv.gz --out charts` | `charts/` |
| 6. RFM segmentation | `python retail_rfm.py prepared/retail_prepared.csv.gz --out rfm` | `rfm/` |
| 7. Cancellation and market evidence | `python retail_part6_evidence.py --out part6` | `part6/` |
| 8. Compact figures | `python make_figures.py` | `figures/` |

```bash
pip install -r requirements.txt
```

Steps 3–8 check that the cleaning flags are present and that revenue totals reconcile before they run.

## Data

- `data/online_retail.xlsx` — raw dataset.
- `prepared/retail_prepared.csv.gz` — original 8 columns plus cleaning flags (line type, product class, duplicate evidence, cancellation status, extreme-quantity reason, analysis filters).
- Other folders — the output tables produced by each step.

## Data source and licence

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/352/online+retail. Licensed under CC BY 4.0.
