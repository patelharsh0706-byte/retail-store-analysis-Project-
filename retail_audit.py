#!/usr/bin/env python3
"""
Online Retail data-quality audit (read-only).

Flags possible data-quality issues in an Online Retail transaction file
(InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
CustomerID, Country). It does NOT clean, drop, or modify anything: the raw
data is loaded once, never mutated, and every finding is written out as a
boolean flag next to the original row so an analyst can decide what to do.

Usage
-----
    python retail_audit.py path/to/Online_Retail.xlsx --out audit_output

Outputs (in --out)
------------------
    audit_summary.csv      one row per check: rows affected, % rows, gross
                           revenue touched, severity, and the analytical impact
    audit_report.md        human-readable report with the same content plus
                           supporting tables
    row_flags.csv.gz       every raw row + one boolean column per row-level check
    examples/<check>.csv   up to N example rows per check, for eyeballing
    run_metadata.json      file hash, row count, library versions, parameters

Reproducibility
---------------
* No randomness; examples are the first N flagged rows in file order.
* All thresholds live in CONFIG below and are written to run_metadata.json.
* Input SHA-256 is recorded so results can be tied to an exact file.
* Identifier columns are read as strings so leading zeros / letter suffixes
  are preserved exactly as stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration — every threshold used by the audit
# --------------------------------------------------------------------------
CONFIG = {
    # StockCodes that are not merchandise (fees, postage, manual adjustments).
    # Matched case-insensitively after stripping whitespace.
    "non_product_codes": [
        "POST", "DOT", "C2", "M", "D", "S", "B", "BANK CHARGES", "AMAZONFEE",
        "CRUK", "PADS", "ADJUST", "ADJUST2", "TEST001", "TEST002",
    ],
    # Patterns that also indicate non-merchandise codes (regex, case-insensitive)
    "non_product_code_patterns": [r"^gift_", r"^DCGS", r"^TEST"],
    # Within-product outliers: a line is flagged when its value is more than
    # RATIO x (or less than 1/RATIO x) the median for that StockCode. Ratios are
    # used instead of z-scores because prices sit on discrete tiers
    # (e.g. 2.08 vs 4.13), which makes MAD/std-based tests over-flag.
    "price_ratio_to_median": 5.0,
    "qty_ratio_to_median": 50.0,
    "qty_outlier_min_units": 100,
    "outlier_min_group_n": 20,
    # Absolute "eyeball" thresholds for extreme single lines
    "extreme_quantity_abs": 10_000,
    "extreme_unit_price": 5_000,
    # A StockCode whose max/min positive unit price ratio exceeds this
    "price_ratio_threshold": 10.0,
    # Keywords in Description that suggest stock adjustments, not sales
    "adjustment_keywords": [
        "damage", "lost", "wet", "mouldy", "missing", "broken", "crushed",
        "faulty", "smashed", "sold as", "found", "check", "adjust", "wrong",
        "mix up", "mixed up", "dotcom", "amazon", "ebay", "returned",
        "sample", "rusty", "thrown away", "unsaleable", "incorrect", "error",
        r"\?",
    ],
    # Country labels that are not a real single country
    "ambiguous_countries": ["Unspecified", "European Community"],
    "n_examples": 25,
}

EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


# --------------------------------------------------------------------------
# Check registry
# --------------------------------------------------------------------------
@dataclass
class Finding:
    check_id: str
    title: str
    severity: str            # High / Medium / Low
    affects: str             # Revenue / Customer / Product (comma separated)
    impact: str              # how it distorts analysis
    mask: pd.Series | None = None   # row-level flag (None for table-level checks)
    n_items: int | None = None      # count for non-row-level checks
    detail: pd.DataFrame | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> pd.DataFrame:
    """Load without coercion of identifiers. Nothing is dropped or fixed."""
    dtype = {"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=dtype)
    else:
        df = pd.read_csv(path, dtype=dtype, encoding_errors="replace")
    df.insert(0, "source_row", np.arange(2, len(df) + 2))  # Excel row number (header = 1)
    return df


def ratio_to_group_median(values: pd.Series, groups: pd.Series, min_n: int) -> pd.Series:
    """value / median(value) within its group; NaN where the group has < min_n rows."""
    v = values.astype(float)
    g = groups.astype(str)
    med = v.groupby(g).transform("median")
    n = v.groupby(g).transform("count")
    r = v / med
    return r.where(n >= min_n)


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def run_checks(raw: pd.DataFrame) -> tuple[list[Finding], dict]:
    df = raw  # alias, read-only by convention: we only build derived Series
    c = CONFIG
    findings: list[Finding] = []
    ctx: dict = {}

    inv = df["InvoiceNo"].astype(str).str.strip()
    code = df["StockCode"].astype(str).str.strip()
    code_u = code.str.upper()
    desc = df["Description"]
    qty = df["Quantity"]
    price = df["UnitPrice"]
    cust = df["CustomerID"]
    line_rev = qty * price  # audit metric only; never written back to raw
    ctx["gross_line_revenue"] = float(line_rev.sum())
    ctx["line_rev"] = line_rev

    is_cancel_inv = inv.str.upper().str.startswith("C")
    non_product_re = "|".join(c["non_product_code_patterns"])
    is_non_product = code_u.isin([x.upper() for x in c["non_product_codes"]]) | \
        code.str.contains(non_product_re, case=False, regex=True, na=False)

    # ---- 0. Schema --------------------------------------------------------
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS + ["source_row"]]
    dtypes = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    findings.append(Finding(
        "schema", "Schema and data types", "Low", "Revenue, Customer, Product",
        "CustomerID is stored as a float (17850.0) and InvoiceNo/StockCode mix "
        "numbers and text. Joining to a CRM or product master on these keys can "
        "silently fail ('17850' != '17850.0'), and Excel round-trips can strip "
        "letter suffixes or leading zeros, merging distinct products.",
        n_items=len(missing_cols) + len(extra_cols), detail=dtypes,
        note=f"missing columns: {missing_cols or 'none'}; unexpected: {extra_cols or 'none'}; "
             f"CustomerID dtype = {df['CustomerID'].dtype}",
    ))

    # ---- 1. Missing CustomerID ----------------------------------------------
    m = cust.isna()
    med_with, med_without = price[~m].median(), price[m].median()
    findings.append(Finding(
        "missing_customer_id", "Missing CustomerID", "High", "Customer, Revenue",
        "About a quarter of lines have no customer. Customer counts, RFM, CLV, "
        "retention and cohort analyses are computed on a biased subset (guest / "
        "walk-in / manual orders behave differently). If these rows are dropped "
        "for customer work but kept for revenue, the two views will not reconcile.",
        mask=m,
        note=f"median UnitPrice is {med_without:.2f} on rows without CustomerID vs "
             f"{med_with:.2f} with one, so these rows are not a random sample",
    ))

    # ---- 2. Missing / blank Description -------------------------------------
    m = desc.isna() | desc.astype(str).str.strip().eq("")
    findings.append(Finding(
        "missing_description", "Missing or blank Description", "Medium", "Product",
        "Lines without a description cannot be named in product reports and are "
        "mostly zero-price stock adjustments, not sales. Grouping by Description "
        "would create an 'NaN' product; grouping by StockCode keeps them but mixes "
        "adjustments into the product's unit history.",
        mask=m,
        note=f"{int((m & (price == 0)).sum()):,} of these have UnitPrice = 0 and "
             f"{int((m & cust.isna()).sum()):,} have no CustomerID",
    ))

    # ---- 3. Exact duplicate rows --------------------------------------------
    m = df.drop(columns="source_row").duplicated(keep="first")
    findings.append(Finding(
        "exact_duplicates", "Exact duplicate rows (all 8 fields identical)", "High",
        "Revenue, Product, Customer",
        "Identical lines inflate revenue, units and order frequency. They may be "
        "legitimate (same item scanned twice on one invoice) or an extract/ETL "
        "duplication — the data cannot tell you which, so confirm with the source "
        "system before removing. Only the 2nd+ occurrence is flagged.",
        mask=m,
    ))

    # ---- 4. Cancellation invoices -------------------------------------------
    findings.append(Finding(
        "cancellation_invoice", "Cancellation invoices (InvoiceNo starts with 'C')",
        "High", "Revenue, Customer, Product",
        "Cancellations carry negative quantities. Summing all lines gives net "
        "revenue; filtering Quantity > 0 gives gross sales but leaves cancelled "
        "orders counted as sales. Either way, treat returns explicitly: they "
        "lower product sell-through, can make a customer look more or less "
        "valuable, and inflate order counts if cancellation invoices are counted "
        "as orders.",
        mask=is_cancel_inv,
    ))

    # ---- 5. Sign inconsistencies --------------------------------------------
    m = (qty < 0) & ~is_cancel_inv
    findings.append(Finding(
        "negative_qty_not_cancelled", "Negative quantity on a non-cancellation invoice",
        "Medium", "Product, Revenue",
        "These are usually warehouse write-offs (damaged, lost, 'wet') entered as "
        "sales lines, typically at price 0 and with no customer. They don't move "
        "revenue much but they reduce unit volumes and can make a product look "
        "like it sold less, or even net negative in a month.",
        mask=m,
    ))
    m = (qty > 0) & is_cancel_inv
    findings.append(Finding(
        "positive_qty_on_cancellation", "Positive quantity on a cancellation invoice",
        "Medium", "Revenue",
        "A cancellation that adds units contradicts the invoice type and would "
        "double-count revenue if cancellations are netted by sign.",
        mask=m,
    ))

    # ---- 6. Cancellations with no prior matching purchase -------------------
    # Heuristic: for each cancel line, is there an earlier-or-same-time purchase
    # by the same customer of the same StockCode within the file?
    purch = df.loc[~is_cancel_inv & (qty > 0) & cust.notna(), ["CustomerID", "StockCode", "InvoiceDate"]]
    first_buy = purch.assign(StockCode=purch["StockCode"].astype(str).str.strip()) \
        .groupby(["CustomerID", "StockCode"])["InvoiceDate"].min()
    canc = df.loc[is_cancel_inv & cust.notna(), ["CustomerID", "InvoiceDate"]].copy()
    canc["StockCode"] = code[canc.index]
    key = pd.MultiIndex.from_frame(canc[["CustomerID", "StockCode"]])
    fb = first_buy.reindex(key).to_numpy()
    unmatched = pd.isna(fb) | (fb > canc["InvoiceDate"].to_numpy())
    m = pd.Series(False, index=df.index)
    m.loc[canc.index[unmatched]] = True
    findings.append(Finding(
        "orphan_cancellation", "Cancellation with no earlier matching purchase in file",
        "Medium", "Revenue, Customer",
        "The original sale is likely before the data window starts. Netting these "
        "against in-window sales understates revenue for the first months and can "
        "make a customer's net spend negative. Heuristic match on CustomerID + "
        "StockCode + earlier date; non-product codes (e.g. 'D' discounts) are "
        "included here and also flagged separately.",
        mask=m,
    ))

    # ---- 7. Prices ----------------------------------------------------------
    findings.append(Finding(
        "zero_price", "UnitPrice = 0", "Medium", "Product, Revenue",
        "Zero-price lines are free samples, adjustments or data-entry gaps. They "
        "add units with no revenue, pulling average selling price down and "
        "inflating product 'popularity' and basket-size metrics.",
        mask=price == 0,
    ))
    findings.append(Finding(
        "negative_price", "UnitPrice < 0", "High", "Revenue",
        "Negative prices are accounting entries (e.g. 'Adjust bad debt'), not "
        "sales. A single line can move monthly revenue by thousands.",
        mask=price < 0,
    ))

    # ---- 8. Non-product StockCodes -----------------------------------------
    findings.append(Finding(
        "non_product_code", "Non-merchandise StockCode (postage, fees, manual, discounts)",
        "High", "Revenue, Product",
        "POST, DOT, M (Manual), D (Discount), BANK CHARGES, AMAZONFEE, CRUK etc. "
        "are not products. Left in, POSTAGE/DOTCOM POSTAGE rank among the "
        "'top products'; Manual and AMAZONFEE lines contain the largest single "
        "values in the file and swing monthly revenue; per-product price and "
        "margin analysis becomes meaningless for these codes.",
        mask=is_non_product,
        detail=df.loc[is_non_product].assign(StockCode=code_u[is_non_product], rev=line_rev[is_non_product])
            .groupby("StockCode").agg(rows=("rev", "size"), net_revenue=("rev", "sum"),
                                      example_description=("Description", "first"))
            .sort_values("rows", ascending=False).reset_index(),
    ))

    # ---- 9. Stock-adjustment descriptions -----------------------------------
    kw = "|".join(c["adjustment_keywords"])
    m = desc.astype(str).str.contains(kw, case=False, regex=True, na=False) & \
        ((price == 0) | (qty < 0)) & ~is_cancel_inv
    findings.append(Finding(
        "adjustment_description", "Description looks like a stock note (damaged, lost, '?')",
        "Low", "Product",
        "Warehouse notes typed into Description ('damaged', 'wet pallet', '?') "
        "create fake product names and distort unit volumes for the real "
        "StockCode. Filter on (price = 0 or qty < 0) to avoid matching genuine "
        "products whose name contains a keyword.",
        mask=m,
    ))

    # ---- 10. Extreme single lines -----------------------------------------
    m = (qty.abs() >= c["extreme_quantity_abs"]) | (price.abs() >= c["extreme_unit_price"])
    findings.append(Finding(
        "extreme_values", f"Extreme line (abs Quantity >= {c['extreme_quantity_abs']:,} or "
                          f"abs UnitPrice >= {c['extreme_unit_price']:,})",
        "High", "Revenue, Customer, Product",
        "A handful of lines (e.g. 80,995 units of one item, cancelled minutes "
        "later) dominate totals: they create a fake revenue spike in one month, "
        "make one customer the 'top customer', and make a cheap item the 'best "
        "seller' unless the matching cancellation is also netted.",
        mask=m,
    ))
    sale = (qty > 0) & ~is_cancel_inv & ~is_non_product
    qr = ratio_to_group_median(qty.where(sale), code_u, c["outlier_min_group_n"])
    m = sale & (qr > c["qty_ratio_to_median"]) & (qty >= c["qty_outlier_min_units"])
    findings.append(Finding(
        "qty_outlier_within_product",
        f"Quantity > {c['qty_ratio_to_median']:g}x the product's median order size (and >= {c['qty_outlier_min_units']} units)",
        "Low", "Product, Customer",
        "Unusually large orders relative to the product's normal order size are "
        "often wholesale orders; they are real but skew mean order size, "
        "forecasting and customer segmentation. Review before excluding.",
        mask=m,
    ))
    psale = (price > 0) & ~is_non_product
    pr_ = ratio_to_group_median(price.where(psale), code_u, c["outlier_min_group_n"])
    k = c["price_ratio_to_median"]
    m = psale & ((pr_ > k) | (pr_ < 1 / k))
    findings.append(Finding(
        "price_outlier_within_product",
        f"UnitPrice > {k:g}x or < 1/{k:g} of the product's median price",
        "Medium", "Revenue, Product",
        "A price far from the product's usual price suggests a keying error, a "
        "pack-vs-unit mismatch or a manual override. Average selling price and "
        "price-elasticity estimates are sensitive to these.",
        mask=m,
    ))

    # ---- 11. StockCode / Description consistency ----------------------------
    d_clean = desc.astype(str).str.strip()
    per_code = pd.DataFrame({"code": code_u, "desc": d_clean})[desc.notna()]
    n_desc = per_code.groupby("code")["desc"].nunique()
    multi = n_desc[n_desc > 1].sort_values(ascending=False)
    findings.append(Finding(
        "code_multiple_descriptions", "StockCode with more than one Description", "Medium", "Product",
        "Product reports grouped by Description split one product into several "
        "rows (and merge different products sharing a name, below). Pick one "
        "canonical key — StockCode — and a rule for the display name.",
        n_items=len(multi),
        detail=per_code[per_code["code"].isin(multi.index[:30])]
            .groupby("code")["desc"].agg(lambda s: " | ".join(sorted(set(s))[:6]))
            .reindex(multi.index[:30]).rename("descriptions").reset_index()
            .assign(n_descriptions=multi.values[:30]),
    ))
    n_code = per_code.groupby("desc")["code"].nunique()
    multi_c = n_code[n_code > 1].sort_values(ascending=False)
    findings.append(Finding(
        "description_multiple_codes", "Description shared by more than one StockCode", "Low", "Product",
        "The same name on several codes (colour/size variants or re-coded items) "
        "means a Description-level report merges distinct SKUs, while a "
        "StockCode-level report splits what the business may consider one product.",
        n_items=len(multi_c),
        detail=multi_c.head(30).rename("n_stockcodes").reset_index(),
    ))
    case_variants = code.groupby(code_u).nunique()
    case_variants = case_variants[case_variants > 1]
    # flag only the minority spelling (the one containing lowercase letters)
    m = code_u.isin(case_variants.index) & code.str.contains("[a-z]", regex=True)
    findings.append(Finding(
        "stockcode_case_variants", "StockCode differs only by letter case (lowercase variant flagged)",
        "Low", "Product",
        "Codes like '85123a' and '85123A' are counted as two products in any "
        "case-sensitive groupby, splitting sales and stock history.",
        mask=m, note=f"{len(case_variants):,} StockCodes have case variants",
    ))
    ws = desc.notna() & (desc.astype(str) != desc.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
    findings.append(Finding(
        "description_whitespace", "Description with leading/trailing/double spaces", "Low", "Product",
        "Whitespace variants of the same name group separately, inflating the "
        "product count and splitting volumes.",
        mask=ws,
        note=f"{int((ws & (desc.astype(str) != desc.astype(str).str.strip())).sum()):,} have trailing/leading "
             "spaces (harmless unless names are joined to another table); "
             f"{int((ws & desc.astype(str).str.contains('  ', regex=False)).sum()):,} contain double spaces",
    ))

    # ---- 12. Price dispersion per product ----------------------------------
    pos = pd.DataFrame({"code": code_u, "price": price})[(price > 0) & ~is_non_product]
    pr = pos.groupby("code")["price"].agg(["min", "median", "max", "nunique", "size"])
    pr["max_min_ratio"] = pr["max"] / pr["min"]
    wide = pr[pr["max_min_ratio"] > c["price_ratio_threshold"]].sort_values("max_min_ratio", ascending=False)
    findings.append(Finding(
        "wide_price_range", f"Product with max/min unit price ratio > {c['price_ratio_threshold']:g}",
        "Medium", "Revenue, Product",
        "Large price spread for one SKU usually signals mixed units of sale "
        "(single vs pack), wholesale vs retail pricing, or entry errors. Revenue "
        "is still 'right' line by line, but average price, price-mix and "
        "revenue-per-unit comparisons across products are not.",
        n_items=len(wide), detail=wide.head(30).reset_index(),
    ))

    # ---- 13. Invoice-level consistency --------------------------------------
    g = df.assign(_inv=inv).groupby("_inv")
    inv_stats = pd.DataFrame({
        "n_customers": g["CustomerID"].nunique(dropna=False),
        "n_dates": g["InvoiceDate"].nunique(),
        "n_countries": g["Country"].nunique(),
    })
    for col, title, imp in [
        ("n_customers", "Invoice linked to more than one CustomerID (incl. missing)",
         "An invoice should belong to one customer. Mixed IDs mean revenue for that "
         "order is split across customers in CLV/RFM."),
        ("n_dates", "Invoice with more than one InvoiceDate",
         "Lines of one order carry different timestamps; daily/hourly sales and "
         "order-level date logic (first purchase, recency) depend on which line you pick."),
        ("n_countries", "Invoice with more than one Country",
         "Country revenue splits a single order across markets."),
    ]:
        bad = inv_stats.index[inv_stats[col] > 1]
        findings.append(Finding(
            f"invoice_{col}", title, "Low", "Revenue, Customer", imp,
            mask=inv.isin(bad), note=f"{len(bad):,} invoices affected",
        ))
    # Same StockCode repeated on one invoice (not exact duplicate)
    dup_line = df.assign(_inv=inv, _code=code_u).duplicated(["_inv", "_code"], keep=False) & \
        ~df.drop(columns="source_row").duplicated(keep=False)
    findings.append(Finding(
        "repeated_item_on_invoice", "Same StockCode on multiple lines of one invoice (not exact duplicate)",
        "Low", "Product",
        "Split lines (different qty or price) are usually legitimate, but they "
        "inflate 'line count' / 'items per basket' metrics and complicate "
        "matching cancellations back to the original line.",
        mask=dup_line,
    ))

    # ---- 14. Customer-level consistency -------------------------------------
    cc = df[cust.notna()].groupby("CustomerID")["Country"].nunique()
    multi_country = cc[cc > 1].index
    findings.append(Finding(
        "customer_multiple_countries", "CustomerID appearing with more than one Country",
        "Low", "Customer",
        "A customer assigned to several countries is double-counted in country-"
        "level customer counts and their revenue is split across markets.",
        mask=cust.isin(multi_country), note=f"{len(multi_country):,} customers affected",
    ))
    net_by_cust = line_rev[cust.notna()].groupby(cust[cust.notna()]).sum()
    neg_cust = net_by_cust[net_by_cust < 0]
    findings.append(Finding(
        "customer_negative_net", "Customer with negative net revenue over the whole file",
        "Medium", "Customer, Revenue",
        "Usually customers whose original purchases pre-date the file (see "
        "orphan cancellations). They break CLV models that assume non-negative "
        "spend and distort segment averages.",
        n_items=len(neg_cust), detail=neg_cust.sort_values().head(30).rename("net_revenue").reset_index(),
    ))

    # ---- 15. Country labels -------------------------------------------------
    m = df["Country"].isin(c["ambiguous_countries"])
    findings.append(Finding(
        "ambiguous_country", "Country is 'Unspecified' or 'European Community'",
        "Low", "Customer, Revenue",
        "These are not countries; geographic revenue and customer counts will "
        "have an unassignable bucket. Also note 'EIRE' (Ireland) and 'RSA' "
        "(South Africa) use non-standard names that won't join to ISO country lists.",
        mask=m,
        detail=df["Country"].value_counts().rename("rows").reset_index(),
    ))

    # ---- 16. Time coverage --------------------------------------------------
    dts = df["InvoiceDate"]
    monthly = df.assign(month=dts.dt.to_period("M").astype(str), rev=line_rev) \
        .groupby("month").agg(rows=("rev", "size"), days_with_sales=("InvoiceDate", lambda s: s.dt.date.nunique()),
                              first=("InvoiceDate", "min"), last=("InvoiceDate", "max"),
                              net_revenue=("rev", "sum")).reset_index()
    first_m, last_m = monthly.iloc[0], monthly.iloc[-1]
    all_days = pd.date_range(dts.min().normalize(), dts.max().normalize(), freq="D")
    missing_days = all_days.difference(pd.DatetimeIndex(dts.dt.normalize().unique()))
    findings.append(Finding(
        "partial_periods", "Partial first/last month and calendar gaps", "High", "Revenue, Customer",
        f"Data runs {dts.min():%Y-%m-%d} to {dts.max():%Y-%m-%d}. The last month "
        f"({last_m['month']}) has only {last_m['days_with_sales']} trading days, so a "
        "naive monthly chart shows a collapse that isn't real; MoM/YoY and "
        "customer churn ('no purchase in last 30 days') are biased at the end. "
        "Only one December appears in full, so seasonality cannot be estimated "
        "from a single cycle. There are no Saturday transactions at all, and "
        "the Christmas/New Year shutdown leaves a multi-week gap; both matter "
        "for daily averages and day-of-week analysis.",
        n_items=len(missing_days), detail=monthly,
        note=f"{len(missing_days)} calendar days with no transactions; "
             f"weekday counts: {dts.dt.day_name().value_counts().to_dict()}",
    ))

    # ---- 17. Missing values everywhere -------------------------------------
    na = raw.drop(columns="source_row").isna().sum().rename("missing").to_frame()
    na["pct"] = (na["missing"] / len(raw) * 100).round(2)
    ctx["missing_table"] = na.reset_index().rename(columns={"index": "column"})

    return findings, ctx


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarise(findings: list[Finding], raw: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    n = len(raw)
    line_rev = ctx["line_rev"]
    rows = []
    for f in findings:
        if f.mask is not None:
            k = int(f.mask.sum())
            rev = line_rev[f.mask]
            rows.append(dict(check_id=f.check_id, title=f.title, severity=f.severity, affects=f.affects,
                             level="row", rows_flagged=k, pct_rows=round(k / n * 100, 3),
                             net_revenue_of_flagged=round(float(rev.sum()), 2),
                             abs_revenue_of_flagged=round(float(rev.abs().sum()), 2),
                             items_flagged=None, note=f.note, impact=f.impact))
        else:
            rows.append(dict(check_id=f.check_id, title=f.title, severity=f.severity, affects=f.affects,
                             level="table", rows_flagged=None, pct_rows=None,
                             net_revenue_of_flagged=None, abs_revenue_of_flagged=None,
                             items_flagged=f.n_items, note=f.note, impact=f.impact))
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    d = df.head(max_rows).copy()

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer() and abs(v) < 1e12):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.2f}"
        return str(v).replace("|", "\\|")
    for col in d.columns:
        d[col] = d[col].map(fmt)
    return d.to_markdown(index=False, disable_numparse=True)


def write_outputs(findings, summary, raw, ctx, out: Path, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(exist_ok=True)
    summary.to_csv(out / "audit_summary.csv", index=False)

    flags = raw.copy()
    for f in findings:
        if f.mask is not None:
            flags[f"flag_{f.check_id}"] = f.mask.values
            if f.mask.any():
                raw[f.mask.values].head(CONFIG["n_examples"]).to_csv(out / "examples" / f"{f.check_id}.csv", index=False)
        elif f.detail is not None:
            f.detail.to_csv(out / "examples" / f"{f.check_id}.csv", index=False)
    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    flags["n_flags"] = flags[flag_cols].sum(axis=1)
    flags.to_csv(out / "row_flags.csv.gz", index=False, compression="gzip")

    sev_order = {"High": 0, "Medium": 1, "Low": 2}
    gross = ctx["gross_line_revenue"]
    md = [f"# Online Retail data-quality audit\n",
          f"Input: `{meta['input_file']}` (sha256 `{meta['sha256'][:16]}…`)  ",
          f"Rows: {len(raw):,} · Columns: {raw.shape[1]-1} · Run: {meta['run_utc']}  ",
          f"Net line revenue as loaded (sum of Quantity × UnitPrice, all rows): {gross:,.2f}\n",
          "Nothing was removed or modified. Each row-level check is a boolean column in "
          "`row_flags.csv.gz`; checks overlap, so percentages do not add up.\n",
          "## Missing values by column\n", to_md(ctx["missing_table"]), "\n",
          "## Summary\n",
          to_md(summary.assign(_s=summary["severity"].map(sev_order)).sort_values(["_s", "rows_flagged"], ascending=[True, False])
                [["severity", "title", "affects", "rows_flagged", "pct_rows", "net_revenue_of_flagged", "items_flagged"]], 100),
          "\n## Findings\n"]
    for f in sorted(findings, key=lambda x: sev_order[x.severity]):
        s = summary.set_index("check_id").loc[f.check_id]
        md.append(f"### [{f.severity}] {f.title}\n")
        if f.mask is not None:
            md.append(f"Rows flagged: **{int(s.rows_flagged):,}** ({s.pct_rows}% of rows) · "
                      f"net revenue on these rows: {s.net_revenue_of_flagged:,.2f} · "
                      f"absolute: {s.abs_revenue_of_flagged:,.2f}  ")
        else:
            md.append(f"Items flagged: **{f.n_items:,}**  ")
        if f.note:
            md.append(f"Note: {f.note}  ")
        md.append(f"**Affects:** {f.affects}  \n**Why it matters:** {f.impact}\n")
        if f.detail is not None and f.check_id != "schema":
            md.append(to_md(f.detail, 15) + "\n")
        elif f.mask is not None and f.mask.any():
            md.append(to_md(raw[f.mask.values].head(8)) + "\n")
    (out / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Online Retail .xlsx or .csv")
    ap.add_argument("--out", type=Path, default=Path("audit_output"))
    a = ap.parse_args(argv)

    raw = load(a.input)
    snapshot = pd.util.hash_pandas_object(raw, index=True).sum()
    findings, ctx = run_checks(raw)
    # Guarantee the audit was read-only
    assert pd.util.hash_pandas_object(raw, index=True).sum() == snapshot, "raw data was modified"

    summary = summarise(findings, raw, ctx)
    meta = {
        "input_file": a.input.name, "sha256": sha256(a.input), "rows": len(raw),
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "config": CONFIG,
    }
    write_outputs(findings, summary, raw, ctx, a.out, meta)

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(summary[["severity", "check_id", "rows_flagged", "pct_rows", "net_revenue_of_flagged", "items_flagged"]].to_string(index=False))
    print(f"\nWrote report to {a.out.resolve()}")


if __name__ == "__main__":
    main()
