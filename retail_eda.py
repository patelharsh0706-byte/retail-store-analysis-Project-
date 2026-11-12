#!/usr/bin/env python3
"""
Part 3 — Exploratory data analysis, step 1: descriptive statistics.

Two deliberately separate views, both read from retail_prepared.csv.gz
(produced by retail_prepare.py). Neither changes the prepared data.

VIEW 1  Valid merchandise sales  ("normal purchase behaviour")
    in_merchandise_sales & line_type == 'sale'
    = real products, positive quantity, positive price, ordinary invoices.
    Excludes: cancellations, stock adjustments, non-product charges,
    accounting adjustments, free items (price 0 would distort price stats),
    and corrected double entries (pattern A). Exact duplicates are kept
    (no reasonable evidence they are repeat recordings).

VIEW 2  Net revenue  (cancellation-adjusted)
    in_net_merchandise = VIEW 1 + free items + product cancellations
    (matched and unmatched), minus corrected double entries.

Usage:  python retail_eda.py prepared/retail_prepared.csv.gz --out eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FLAGS = [
    "line_type", "is_product", "is_exact_duplicate", "dup_evidence", "dup_removed",
    "possible_repeat_invoice", "repeat_group", "repeat_pattern", "repeat_role",
    "repeat_needs_review", "corrected_double_entry", "cancellation_status",
    "qty_extreme", "qty_extreme_reason", "in_merchandise_sales", "in_net_merchandise",
    "in_customer_analysis",
]
PCTS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


# ------------------------------------------------------------------ checks
def verify(df: pd.DataFrame, recon_path: Path) -> pd.DataFrame:
    """Fail loudly if the prepared file lacks the updated flags or they disagree."""
    missing = [c for c in REQUIRED_FLAGS if c not in df.columns]
    assert not missing, f"prepared data is missing flags: {missing} — rerun retail_prepare.py"
    cde = df["corrected_double_entry"]
    checks = [
        ("All required flag columns present", True, f"{len(REQUIRED_FLAGS)} columns"),
        ("Exact duplicates flagged", df["is_exact_duplicate"].sum() > 0, f"{int(df['is_exact_duplicate'].sum()):,} rows"),
        ("Duplicates removed under agreed rule (expected 0)", df["dup_removed"].sum() == 0, f"{int(df['dup_removed'].sum()):,}"),
        ("Every repeat-invoice row has a pattern",
         df.loc[df["possible_repeat_invoice"], "repeat_pattern"].notna().all(),
         f"{df.loc[df['possible_repeat_invoice'], 'repeat_group'].nunique()} groups"),
        ("Corrected double entries: copies = correcting cancellations",
         (df["repeat_role"] == "extra_copy").sum() == (df["repeat_role"] == "correcting_cancellation").sum(),
         f"{int((df['repeat_role'] == 'extra_copy').sum())} each"),
        ("Corrected double entries net to zero",
         abs(df.loc[cde, "line_revenue"].sum()) < 0.01, f"{df.loc[cde, 'line_revenue'].sum():,.2f}"),
        ("No corrected double entry inside merchandise or net views",
         not (cde & (df["in_merchandise_sales"] | df["in_net_merchandise"])).any(), "0 rows"),
        ("No stock adjustment / non-product row in either view",
         not ((df["in_merchandise_sales"] | df["in_net_merchandise"]) &
              df["line_type"].isin(["stock_adjustment", "non_product_charge", "accounting_adjustment"])).any(), "0 rows"),
        ("Pattern B review rows kept (flagged, not excluded)",
         df.loc[df["repeat_needs_review"], "in_merchandise_sales"].all(), f"{int(df['repeat_needs_review'].sum())} rows"),
    ]
    if recon_path.exists():
        rec = pd.read_csv(recon_path).set_index("step")["category_revenue"]
        net_ok = abs(rec["= Net merchandise revenue"] - df.loc[df["in_net_merchandise"], "line_revenue"].sum()) < 0.01
        gross_ok = abs(rec["= Gross merchandise sales"] - df.loc[df["in_merchandise_sales"], "line_revenue"].sum()) < 0.01
        checks.append(("Net and gross totals match revenue_reconciliation.csv", net_ok and gross_ok,
                       f"net {rec['= Net merchandise revenue']:,.2f} / gross {rec['= Gross merchandise sales']:,.2f}"))
    out = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    assert out["passed"].all(), "\n" + out[~out["passed"]].to_string()
    return out


# ------------------------------------------------------------------ helpers
def describe(s: pd.Series, name: str) -> dict:
    s = s.dropna().astype(float)
    q = s.quantile(PCTS)
    return {"metric": name, "n": len(s), "mean": s.mean(), "std": s.std(), "min": s.min(),
            **{f"p{int(p*100)}": q[p] for p in PCTS}, "max": s.max(),
            "skew": s.skew(), "mean/median": s.mean() / q[0.5] if q[0.5] else np.nan}


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]) and c != "n":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else
                            (f"{v:.3g}" if 0 < abs(v) < 0.1 else f"{v + 0.0:,.2f}")))
        elif c == "n":
            d[c] = d[c].map(lambda v: f"{int(v):,}")
    return d.to_markdown(index=False, disable_numparse=True)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eda"))
    a = ap.parse_args(argv)
    df = pd.read_csv(a.prepared, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"], low_memory=False)
    checks = verify(df, a.prepared.parent / "revenue_reconciliation.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    df["month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # ================= VIEW 1: valid merchandise sales =================
    v1 = df[df["in_merchandise_sales"] & df["line_type"].eq("sale")].copy()
    scope1 = pd.DataFrame([
        ("Line items", len(v1)), ("Invoices (orders)", v1["InvoiceNo"].nunique()),
        ("Customers (with ID)", v1["CustomerID"].nunique()), ("Products (StockCodes)", v1["StockCode"].str.upper().nunique()),
        ("Countries", v1["Country"].nunique()),
        ("First / last date", f"{v1['InvoiceDate'].min():%Y-%m-%d} / {v1['InvoiceDate'].max():%Y-%m-%d}"),
        ("Gross revenue", f"{v1['line_revenue'].sum():,.2f}"),
        ("Lines without CustomerID", f"{v1['CustomerID'].isna().sum():,} ({v1['CustomerID'].isna().mean():.1%})"),
        ("Exact-duplicate lines kept", f"{int(v1['is_exact_duplicate'].sum()):,}"),
        ("Extreme-quantity lines kept", f"{int(v1['qty_extreme'].sum()):,}"),
        ("Pattern B review lines kept", f"{int(v1['repeat_needs_review'].sum()):,}"),
    ], columns=["item", "value"])

    line_stats = pd.DataFrame([describe(v1["Quantity"], "Quantity per line"),
                               describe(v1["UnitPrice"], "Unit price (£)"),
                               describe(v1["line_revenue"], "Line revenue (£)")])
    inv = v1.groupby("InvoiceNo").agg(value=("line_revenue", "sum"), lines=("StockCode", "size"),
                                      units=("Quantity", "sum"), products=("StockCode", "nunique"),
                                      cust=("CustomerID", "first"))
    order_stats = pd.DataFrame([describe(inv["value"], "Order value (£)"),
                                describe(inv["lines"], "Lines per order"),
                                describe(inv["products"], "Distinct products per order"),
                                describe(inv["units"], "Units per order")])
    cu = v1[v1["CustomerID"].notna()].groupby("CustomerID").agg(
        revenue=("line_revenue", "sum"), orders=("InvoiceNo", "nunique"),
        first=("InvoiceDate", "min"), last=("InvoiceDate", "max"))
    cu["aov"] = cu["revenue"] / cu["orders"]
    cust_stats = pd.DataFrame([describe(cu["revenue"], "Gross revenue per customer (£)"),
                               describe(cu["orders"], "Orders per customer"),
                               describe(cu["aov"], "Average order value per customer (£)")])
    # sensitivity: how much do extreme-quantity lines move the line/order means?
    v1x = v1[~v1["qty_extreme"]]
    invx = v1x.groupby("InvoiceNo")["line_revenue"].sum()
    sens = pd.DataFrame([
        ("Mean quantity per line", v1["Quantity"].mean(), v1x["Quantity"].mean()),
        ("Mean line revenue (£)", v1["line_revenue"].mean(), v1x["line_revenue"].mean()),
        ("Mean order value (£)", inv["value"].mean(), invx.mean()),
        ("Median order value (£)", inv["value"].median(), invx.median()),
        ("Gross revenue (£)", v1["line_revenue"].sum(), v1x["line_revenue"].sum()),
    ], columns=["metric", "all valid sales", "excluding extreme-quantity lines"])
    sens["change"] = sens["excluding extreme-quantity lines"] / sens["all valid sales"] - 1

    # ================= VIEW 2: net revenue (cancellation-adjusted) =================
    v2 = df[df["in_net_merchandise"]].copy()
    v2["kind"] = np.where(v2["line_type"].eq("cancellation"),
                          "cancel_" + v2["cancellation_status"].astype(str), v2["line_type"])
    monthly = v2.pivot_table(index="month", columns="kind", values="line_revenue", aggfunc="sum", fill_value=0)
    for c in ["sale", "free_item", "cancel_matched", "cancel_unmatched"]:
        if c not in monthly:
            monthly[c] = 0.0
    monthly = monthly[["sale", "cancel_matched", "cancel_unmatched"]].rename(columns={
        "sale": "gross_sales", "cancel_matched": "matched_cancellations", "cancel_unmatched": "unmatched_cancellations"})
    monthly["net_revenue"] = monthly.sum(axis=1)
    monthly["cancellation_rate"] = -(monthly["matched_cancellations"] + monthly["unmatched_cancellations"]) / monthly["gross_sales"]
    days = v2.groupby("month")["InvoiceDate"].apply(lambda s: s.dt.date.nunique()).rename("trading_days")
    monthly = monthly.join(days)
    monthly["net_per_trading_day"] = monthly["net_revenue"] / monthly["trading_days"]
    tot = monthly[["gross_sales", "matched_cancellations", "unmatched_cancellations", "net_revenue"]].sum()
    assert abs(tot["net_revenue"] - v2["line_revenue"].sum()) < 0.01

    net_summary = pd.DataFrame([
        ("Gross merchandise sales (lines with price > 0)", tot["gross_sales"]),
        ("Matched product cancellations", tot["matched_cancellations"]),
        ("Unmatched product cancellations", tot["unmatched_cancellations"]),
        ("Free items", v2.loc[v2["line_type"].eq("free_item"), "line_revenue"].sum()),
        ("Net merchandise revenue", v2["line_revenue"].sum()),
        ("Cancellation rate (by value)", -(tot["matched_cancellations"] + tot["unmatched_cancellations"]) / tot["gross_sales"]),
        ("Cancellation lines", int(v2["line_type"].eq("cancellation").sum())),
        ("Invoices that are cancellations", v2.loc[v2["line_type"].eq("cancellation"), "InvoiceNo"].nunique()),
    ], columns=["item", "value"])

    nc = v2[v2["CustomerID"].notna()].groupby("CustomerID").agg(
        gross=("line_revenue", lambda s: s[s > 0].sum()), cancelled=("line_revenue", lambda s: s[s < 0].sum()),
        net=("line_revenue", "sum"))
    nc["cancel_share"] = (-nc["cancelled"] / nc["gross"].replace(0, np.nan)) + 0.0   # +0.0 drops -0.0
    net_cust_stats = pd.DataFrame([describe(nc["net"], "Net revenue per customer (£)"),
                                   describe(nc["cancel_share"] * 100, "% of customer's gross cancelled")])
    net_cust_flags = pd.DataFrame([
        ("Customers in net view", len(nc)),
        ("Customers who cancelled at least once", int((nc["cancelled"] < 0).sum())),
        ("Customers with net revenue < 0", int((nc["net"] < 0).sum())),
        ("Customers with net revenue = 0", int((nc["net"].abs() < 0.005).sum())),
        ("Customers who only cancelled (no sale in file)", int((nc["gross"] == 0).sum())),
    ], columns=["item", "value"])

    # ------------------------------------------------------------ write
    for name, t in [("checks", checks), ("v1_scope", scope1), ("v1_line_stats", line_stats),
                    ("v1_order_stats", order_stats), ("v1_customer_stats", cust_stats),
                    ("v1_extreme_sensitivity", sens), ("v2_net_summary", net_summary),
                    ("v2_monthly", monthly.reset_index()), ("v2_customer_net_stats", net_cust_stats),
                    ("v2_customer_flags", net_cust_flags)]:
        t.to_csv(a.out / f"{name}.csv", index=False)

    def kv(t):
        d = t.copy()
        def f(item, v):
            if isinstance(v, str):
                return v
            if "rate" in item.lower():
                return f"{v:.1%}"
            if item.split()[0] in {"Customers", "Invoices", "Cancellation"} and float(v).is_integer():
                return f"{int(v):,}"
            return f"{v:,.2f}"
        d["value"] = [f(i, v) for i, v in zip(d["item"], d["value"])]
        return d.to_markdown(index=False, disable_numparse=True)
    ms = monthly.reset_index().copy()
    ms["cancellation_rate"] = ms["cancellation_rate"].map("{:.1%}".format)
    sens_md = sens.copy(); sens_md["change"] = sens_md["change"].map("{:+.1%}".format)
    md = [
        "# EDA step 1 — descriptive statistics\n",
        "## Pre-flight: prepared data carries the updated flags\n", checks.to_markdown(index=False), "\n",
        "## View 1 — valid merchandise sales (normal purchase behaviour)\n",
        "Definition: `in_merchandise_sales & line_type == 'sale'` — real products, quantity > 0, price > 0, "
        "ordinary invoices. Excludes cancellations, stock adjustments, non-product charges, bad-debt entries, "
        "free items and corrected double entries.\n",
        kv(scope1), "\n", "### Line level\n", fmt(line_stats), "\n",
        "### Order level (one row per invoice)\n", fmt(order_stats), "\n",
        "### Customer level (customers with an ID, gross)\n", fmt(cust_stats), "\n",
        "### Sensitivity: extreme-quantity lines (kept, not yet decided)\n",
        fmt(sens_md.drop(columns="change")).replace("|\n|:", "|\n|:"), "\n",
        "Change when extreme-quantity lines are excluded: " +
        "; ".join(f"{m} {c}" for m, c in zip(sens_md["metric"], sens_md["change"])) + "\n",
        "## View 2 — net revenue (cancellation-adjusted)\n",
        "Definition: `in_net_merchandise` — View 1 + free items + product cancellations (matched and unmatched); "
        "corrected double entries excluded on both sides.\n",
        kv(net_summary), "\n", "### By month\n",
        fmt(ms.drop(columns="cancellation_rate").assign(cancellation_rate=ms["cancellation_rate"])), "\n",
        "### Customer level (net)\n", fmt(net_cust_stats), "\n", kv(net_cust_flags), "\n",
    ]
    (a.out / "eda_descriptive.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
