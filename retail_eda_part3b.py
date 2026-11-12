#!/usr/bin/env python3
"""
IE5005 Part 3 (continued) — frequency tables, Quantity–UnitPrice correlation,
and country comparison. Read-only: no record is modified or removed; every
subset is a filter on the flags written by retail_prepare.py.

Subsets used (same definitions as retail_eda.py):
  ALL     every row of retail_prepared.csv.gz
  VIEW 1  valid merchandise sales: in_merchandise_sales & line_type == 'sale'
  VIEW 2  net merchandise: in_net_merchandise (View 1 + free items + product
          cancellations, corrected double entries excluded on both sides)

Usage:  python retail_eda_part3b.py prepared/retail_prepared.csv.gz --out eda/part3b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retail_eda import verify            # same pre-flight checks as EDA step 1

MIN_ORDERS_FOR_RANKING = 20              # countries below this are shown but marked "small sample"
UK = "United Kingdom"


def pct_table(s: pd.Series, name: str) -> pd.DataFrame:
    t = s.value_counts(dropna=False).rename("count").to_frame()
    t["percent"] = t["count"] / t["count"].sum() * 100
    t["cumulative_percent"] = t["percent"].cumsum()
    return t.rename_axis(name).reset_index()


def corr_row(label, x, y, note=""):
    x, y = np.asarray(x, float), np.asarray(y, float)
    p, pp = stats.pearsonr(x, y)
    s, sp = stats.spearmanr(x, y)
    return {"subset": label, "n": len(x), "pearson_r": p, "pearson_p": pp,
            "spearman_rho": s, "spearman_p": sp, "note": note}


def md(df: pd.DataFrame, floatfmt=",.2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eda/part3b"))
    a = ap.parse_args(argv)
    df = pd.read_csv(a.prepared, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"], low_memory=False)
    n_before, h_before = len(df), pd.util.hash_pandas_object(df, index=True).sum()
    checks = verify(df, a.prepared.parent / "revenue_reconciliation.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    df["code_u"] = df["StockCode"].str.strip().str.upper()   # helper column on the in-memory copy only
    v1 = df[df["in_merchandise_sales"] & df["line_type"].eq("sale")]
    v2 = df[df["in_net_merchandise"]]
    out: dict[str, pd.DataFrame] = {"checks": checks}

    # ===================== 1. Frequency tables =====================
    lt = pct_table(df["line_type"], "line_type")
    lt = lt.merge(df.groupby("line_type")["line_revenue"].sum().rename("revenue").reset_index(), on="line_type")
    lt["revenue_percent_of_total"] = lt["revenue"] / df["line_revenue"].sum() * 100
    out["freq_line_type"] = lt

    c_all = pct_table(df["Country"], "Country").rename(columns={
        "count": "rows_all", "percent": "pct_rows_all", "cumulative_percent": "cum_pct_all"})
    c_v1 = pct_table(v1["Country"], "Country").rename(columns={
        "count": "rows_valid_sales", "percent": "pct_rows_valid_sales", "cumulative_percent": "cum_pct_valid_sales"})
    c_inv = v1.groupby("Country")["InvoiceNo"].nunique().rename("orders").reset_index()
    c_inv["pct_orders"] = c_inv["orders"] / c_inv["orders"].sum() * 100
    country_freq = c_all.merge(c_v1, on="Country", how="left").merge(c_inv, on="Country", how="left").fillna(0)
    out["freq_country"] = country_freq.sort_values("rows_all", ascending=False)

    # ===================== 2. Quantity vs UnitPrice =====================
    q, p = v1["Quantity"], v1["UnitPrice"]
    ext = v1["qty_extreme"]
    top_idx = v1["Quantity"].nlargest(10).index
    rows = [
        corr_row("All valid sales (raw)", q, p),
        corr_row("All valid sales, log10 both", np.log10(q), np.log10(p), "Pearson on log scale ≈ elasticity-style view"),
        corr_row("Excluding extreme-quantity lines", q[~ext], p[~ext], f"{int(ext.sum()):,} lines removed for this test only"),
        corr_row("Excluding the 10 largest lines", q.drop(top_idx), p.drop(top_idx), "influence test"),
        corr_row("Lines with CustomerID", q[v1["CustomerID"].notna()], p[v1["CustomerID"].notna()]),
        corr_row("Lines without CustomerID", q[v1["CustomerID"].isna()], p[v1["CustomerID"].isna()]),
        corr_row("United Kingdom", q[v1["Country"].eq(UK)], p[v1["Country"].eq(UK)]),
        corr_row("Non-UK", q[~v1["Country"].eq(UK)], p[~v1["Country"].eq(UK)]),
    ]
    corr = pd.DataFrame(rows)
    out["corr_qty_price"] = corr

    # Within-product: does the SAME item get a lower price at higher quantity?
    def within(g):
        if len(g) < 30 or g["UnitPrice"].nunique() < 2 or g["Quantity"].nunique() < 2:
            return np.nan
        return stats.spearmanr(g["Quantity"], g["UnitPrice"])[0]
    wp = v1.groupby("code_u")[["Quantity", "UnitPrice"]].apply(within).dropna()
    within_tbl = pd.DataFrame([{
        "products_tested (>=30 lines, varying price)": len(wp),
        "median_within_product_rho": wp.median(), "mean_within_product_rho": wp.mean(),
        "share_negative": (wp < 0).mean() * 100, "share_rho_below_-0.3": (wp < -0.3).mean() * 100,
        "share_positive": (wp > 0).mean() * 100}])
    out["corr_within_product"] = within_tbl
    # Demeaned (within-product) Pearson on log scale: removes the across-product mix effect
    lv = v1.assign(lq=np.log10(q), lp=np.log10(p))
    lv = lv[lv.groupby("code_u")["lq"].transform("size") >= 30]
    dq = lv["lq"] - lv.groupby("code_u")["lq"].transform("mean")
    dp = lv["lp"] - lv.groupby("code_u")["lp"].transform("mean")
    corr_dm = corr_row("Within-product (log, demeaned by StockCode)", dq, dp, "products with >=30 lines")
    out["corr_qty_price"] = pd.concat([corr, pd.DataFrame([corr_dm])], ignore_index=True)

    # Quantity bands: how price and revenue share change with order size
    bands = pd.cut(q, [0, 1, 5, 11, 24, 99, 999, np.inf],
                   labels=["1", "2-5", "6-11", "12-24", "25-99", "100-999", "1000+"])
    qb = v1.groupby(bands, observed=True).agg(lines=("Quantity", "size"), median_unit_price=("UnitPrice", "median"),
                                              mean_unit_price=("UnitPrice", "mean"), units=("Quantity", "sum"),
                                              revenue=("line_revenue", "sum")).reset_index().rename(columns={"Quantity": "qty_band"})
    qb["pct_lines"] = qb["lines"] / qb["lines"].sum() * 100
    qb["pct_revenue"] = qb["revenue"] / qb["revenue"].sum() * 100
    out["qty_bands"] = qb
    top10 = v1.loc[top_idx].assign(CustomerID=lambda d: d["CustomerID"].astype("Int64"))[["InvoiceNo", "StockCode", "Description_std", "Quantity", "UnitPrice", "line_revenue",
                             "CustomerID", "Country", "qty_extreme_reason"]]
    out["largest_lines"] = top10

    # ===================== 3. Country comparison =====================
    g1 = v1.groupby("Country")
    ctry = pd.DataFrame({
        "orders": g1["InvoiceNo"].nunique(),
        "customers": g1["CustomerID"].nunique(),
        "gross_sales": g1["line_revenue"].sum(),
    })
    g2 = v2.groupby("Country")
    ctry["product_cancellations"] = v2[v2["line_type"].eq("cancellation")].groupby("Country")["line_revenue"].sum()
    ctry["net_revenue"] = g2["line_revenue"].sum()
    ctry = ctry.fillna({"product_cancellations": 0.0, "orders": 0, "customers": 0, "gross_sales": 0.0})
    ov = v1.groupby(["Country", "InvoiceNo"])["line_revenue"].sum()
    ctry["aov_mean"] = ctry["gross_sales"] / ctry["orders"].replace(0, np.nan)
    ctry["aov_median"] = ov.groupby("Country").median()
    ctry["net_per_order"] = ctry["net_revenue"] / ctry["orders"].replace(0, np.nan)
    ctry["orders_per_customer"] = ctry["orders"] / ctry["customers"].replace(0, np.nan)
    ctry["cancellation_rate_pct"] = -ctry["product_cancellations"] / ctry["gross_sales"].replace(0, np.nan) * 100 + 0.0
    # concentration: share of a country's net revenue from its largest customer (ID'd customers only)
    cust_net = v2[v2["CustomerID"].notna()].groupby(["Country", "CustomerID"])["line_revenue"].sum()
    ctry["top_customer_share_pct"] = cust_net.groupby("Country").max() / ctry["net_revenue"] * 100
    ctry["pct_net_without_customer_id"] = v2[v2["CustomerID"].isna()].groupby("Country")["line_revenue"].sum() \
        .reindex(ctry.index).fillna(0) / ctry["net_revenue"] * 100
    tot_net = ctry["net_revenue"].sum()
    ctry["pct_of_total_net"] = ctry["net_revenue"] / tot_net * 100
    nonuk_net = ctry.loc[ctry.index != UK, "net_revenue"].sum()
    ctry["pct_of_non_uk_net"] = np.where(ctry.index != UK, ctry["net_revenue"] / nonuk_net * 100, np.nan)
    # extreme-quantity sensitivity: AOV without extreme lines
    v1x = v1[~v1["qty_extreme"]]
    ctry["aov_mean_excl_extreme"] = v1x.groupby("Country")["line_revenue"].sum() / v1x.groupby("Country")["InvoiceNo"].nunique()
    ctry["sample_flag"] = np.where(ctry["orders"] < MIN_ORDERS_FOR_RANKING, f"small sample (<{MIN_ORDERS_FOR_RANKING} orders)", "")
    ctry["label_flag"] = np.where(ctry.index.isin(["Unspecified", "European Community"]), "not a single country", "")
    ctry = ctry.sort_values("net_revenue", ascending=False).reset_index()
    ctry.insert(1, "rank_net", range(1, len(ctry) + 1))
    assert abs(ctry["net_revenue"].sum() - v2["line_revenue"].sum()) < 0.01
    assert int(ctry["orders"].sum()) == v1["InvoiceNo"].nunique()        # each order belongs to one country
    out["country_comparison"] = ctry

    # UK vs non-UK summary
    seg = np.where(v1["Country"].eq(UK), "United Kingdom", "Non-UK")
    seg2 = np.where(v2["Country"].eq(UK), "United Kingdom", "Non-UK")
    s1 = v1.groupby(seg).agg(orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique"),
                             gross_sales=("line_revenue", "sum"), countries=("Country", "nunique"))
    s1["net_revenue"] = v2.groupby(seg2)["line_revenue"].sum()
    s1["pct_net"] = s1["net_revenue"] / s1["net_revenue"].sum() * 100
    s1["pct_orders"] = s1["orders"] / s1["orders"].sum() * 100
    s1["aov_mean"] = s1["gross_sales"] / s1["orders"]
    s1["aov_median"] = v1.assign(seg=seg).groupby(["seg", "InvoiceNo"])["line_revenue"].sum().groupby("seg").median()
    s1["orders_per_customer"] = s1["orders"] / s1["customers"]
    s1["pct_lines_missing_customer"] = v1.assign(seg=seg).groupby("seg")["CustomerID"].apply(lambda x: x.isna().mean() * 100)
    s1["cancellation_rate_pct"] = -v2[v2["line_type"].eq("cancellation")].groupby(
        np.where(v2.loc[v2["line_type"].eq("cancellation"), "Country"].eq(UK), "United Kingdom", "Non-UK"))["line_revenue"].sum() \
        / s1["gross_sales"] * 100
    out["uk_vs_non_uk"] = s1.reset_index(names="segment")
    nonuk = ctry[ctry["Country"] != UK].copy()
    nonuk.insert(1, "rank_non_uk", range(1, len(nonuk) + 1))
    out["non_uk_markets"] = nonuk

    # ------------------------------------------------------------ guarantees
    assert len(df) == n_before
    assert pd.util.hash_pandas_object(df.drop(columns="code_u"), index=True).sum() == h_before, "data modified"
    for k, t in out.items():
        t.to_csv(a.out / f"{k}.csv", index=False)

    # ------------------------------------------------------------ report
    show_c = ["Country", "rank_net", "net_revenue", "pct_of_total_net", "orders", "customers", "aov_mean",
              "aov_median", "aov_mean_excl_extreme", "orders_per_customer", "cancellation_rate_pct",
              "top_customer_share_pct", "sample_flag"]
    rep = [
        "# Part 3 (continued) — frequencies, correlation, country comparison\n",
        "Pre-flight checks passed: " + ", ".join(checks["check"].str.lower()) + ".\n",
        "## 1a. line_type (all rows)\n", md(lt), "\n",
        "## 1b. Country (all rows, valid sales lines, and orders)\n", md(out["freq_country"]), "\n",
        "## 2. Quantity vs UnitPrice (View 1: valid merchandise sales)\n",
        md(out["corr_qty_price"], ".4g"), "\n",
        "Within-product Spearman (same StockCode):\n", md(within_tbl, ".3f"), "\n",
        "Quantity bands:\n", md(qb), "\n", "Ten largest lines:\n", md(top10), "\n",
        "## 3a. UK vs non-UK\n", md(out["uk_vs_non_uk"]), "\n",
        "## 3b. All countries ranked by net merchandise revenue\n", md(ctry[show_c]), "\n",
        "## 3c. Non-UK markets\n", md(nonuk[["Country", "rank_non_uk", "net_revenue", "pct_of_non_uk_net", "orders",
                                          "customers", "aov_mean", "aov_median", "aov_mean_excl_extreme",
                                          "orders_per_customer", "cancellation_rate_pct", "top_customer_share_pct",
                                          "pct_net_without_customer_id", "sample_flag", "label_flag"]]), "\n",
    ]
    (a.out / "part3b_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
