#!/usr/bin/env python3
"""
IE5005 Part 6 — evidence base for recommendations (read-only).

1. Verifies the ten headline figures against the output files of Parts 2–5.
2. Decomposes product cancellations inside net merchandise revenue into
   accounting reversals vs potentially preventable lost sales.
3. Builds market evidence (Germany, France vs concentrated markets) using the
   Part 5 RFM segments.

Usage:  python retail_part6_evidence.py --out part6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retail_eda import verify

UK = "United Kingdom"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=Path("prepared/retail_prepared.csv.gz"))
    ap.add_argument("--out", type=Path, default=Path("part6"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.prepared, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"], low_memory=False)
    h0 = pd.util.hash_pandas_object(df, index=True).sum()
    verify(df, a.prepared.parent / "revenue_reconciliation.csv")

    # ---------------- 1. verification of reported figures ----------------
    rec = pd.read_csv("prepared/revenue_reconciliation.csv").set_index("step")["category_revenue"]
    seg = pd.read_csv("rfm/segment_summary.csv").set_index("segment")
    ctry = pd.read_csv("eda/part3b/country_comparison.csv").set_index("Country")
    v2s = pd.read_csv("eda/v2_net_summary.csv").set_index("item")["value"]
    mon = pd.read_csv("eda/v2_monthly.csv").set_index("month")
    net_total = rec["= Net merchandise revenue"]
    attr = float(pd.read_csv("prepared/revenue_views.csv").set_index("view").loc["Customer-analysis net revenue", "revenue"])
    v = []
    def add(claim, value, source, definition, status, note=""):
        v.append({"claim": claim, "verified_value": value, "source_file": source, "definition": definition,
                  "status": status, "note": note})
    add("Net merchandise revenue £9.79m", f"£{net_total:,.2f}", "prepared/revenue_reconciliation.csv",
        "Product sales + free items + product cancellations (matched & unmatched); excludes non-product charges, "
        "stock & bad-debt adjustments, corrected double entries; includes rows without CustomerID", "Verified")
    for name, claim_n, claim_p in [("Champions", 909, "66%"), ("At-Risk High-Value", 403, "9.1%"), ("Potential Loyalists", 485, "4.9%")]:
        r = seg.loc[name]
        add(f"{name} {claim_n} customers, {claim_p}", f"{int(r['customers'])} customers; {r['pct_of_attributable_net']:.2f}% "
            f"(£{r['net_revenue']:,.2f})", "rfm/segment_summary.csv",
            f"Share of ATTRIBUTABLE net revenue (customers with an ID) = £{attr:,.2f}, not of £9.79m",
            "Verified — base must be stated",
            f"As a share of total net merchandise (£{net_total/1e6:.2f}m) it is {r['net_revenue']/net_total*100:.1f}%")
    uk = ctry.loc[UK]
    add("UK 84.77% of net revenue", f"{uk['pct_of_total_net']:.2f}% (£{uk['net_revenue']:,.2f})",
        "eda/part3b/country_comparison.csv", "Share of total net merchandise revenue (incl. rows without ID)", "Verified")
    for c, cn, cc in [("Germany", "£201k", 94), ("France", "£182k", 87)]:
        r = ctry.loc[c]
        add(f"{c} {cn} and {cc} customers", f"£{r['net_revenue']:,.2f}; {int(r['customers'])} customers",
            "eda/part3b/country_comparison.csv",
            "Net merchandise revenue; customers = IDs with ≥1 valid purchase", "Verified",
            "Germany shows 95 customers in the net view because 1 customer only cancelled" if c == "Germany" else "")
    nl = ctry.loc["Netherlands"]
    add("Netherlands £283k, 98.34% top-customer concentration",
        f"£{nl['net_revenue']:,.2f}; top customer {nl['top_customer_share_pct']:.2f}%",
        "eda/part3b/country_comparison.csv", "Largest customer's net ÷ country net revenue", "Verified")
    add("Cancellation rate 4.6%", f"{float(v2s['Cancellation rate (by value)'])*100:.2f}%",
        "eda/v2_net_summary.csv", "Product cancellations ÷ gross merchandise sales, by value",
        "Verified — but misleading without context",
        "Two fully reversed orders (80,995 and 74,215 units, £245,653) are 51.7% of cancelled value; see recovery table")
    add("November net revenue £1.43m", f"£{mon.loc['2011-11', 'net_revenue']:,.2f}", "eda/v2_monthly.csv",
        "Net merchandise revenue, invoices dated Nov 2011", "Verified", "Single year: peak cannot be called seasonal")
    verif = pd.DataFrame(v)

    # ---------------- 2. cancellation decomposition ----------------
    nm = df[df["in_net_merchandise"]].copy()
    nm["code_u"] = nm["StockCode"].str.strip().str.upper()
    canc = nm[nm["line_type"].eq("cancellation")].copy()
    sales = nm[nm["line_type"].eq("sale") & nm["CustomerID"].notna()][["CustomerID", "code_u", "InvoiceDate", "UnitPrice"]]
    # lag to the most recent earlier purchase of the same item by the same customer
    cc = canc[canc["CustomerID"].notna()].reset_index()[["index", "CustomerID", "code_u", "InvoiceDate"]].sort_values("InvoiceDate")
    ss = sales.rename(columns={"InvoiceDate": "buy_date"}).sort_values("buy_date")
    m = pd.merge_asof(cc, ss, left_on="InvoiceDate", right_on="buy_date", by=["CustomerID", "code_u"], direction="backward")
    lag_h = ((m["InvoiceDate"] - m["buy_date"]).dt.total_seconds() / 3600).to_numpy()
    canc["lag_hours"] = np.nan
    canc.loc[m["index"].to_numpy(), "lag_hours"] = lag_h
    rev_ext = canc["qty_extreme_reason"].eq("cancellation_reversing_extreme_sale")
    cls = np.select(
        [rev_ext,
         canc["CustomerID"].isna(),
         canc["lag_hours"].isna(),
         canc["lag_hours"] <= 24,
         canc["lag_hours"] <= 24 * 30],
        ["A. Full reversal of an extreme-quantity order",
         "E. No CustomerID (cannot be traced)",
         "D. No earlier purchase in file (likely pre-window)",
         "B. Within 24h of purchase",
         "C1. 1–30 days after purchase"],
        "C2. More than 30 days after purchase")
    canc["cancel_class"] = cls
    gross = nm.loc[nm["line_type"].eq("sale"), "line_revenue"].sum()
    canc[["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate", "Quantity", "line_revenue", "lag_hours", "cancel_class"]] \
        .to_csv(a.out / "cancellation_lines_classified.csv", index=False)
    lag_summary = canc.groupby("cancel_class")["lag_hours"].describe(percentiles=[.5]).round(2).reset_index()
    dec = canc.groupby("cancel_class").agg(lines=("line_revenue", "size"), invoices=("InvoiceNo", "nunique"),
                                           customers=("CustomerID", "nunique"), cancelled_value=("line_revenue", "sum"))
    dec["pct_of_cancelled_value"] = dec["cancelled_value"] / canc["line_revenue"].sum() * 100
    dec["pct_of_gross_sales"] = -dec["cancelled_value"] / gross * 100
    dec.loc["TOTAL"] = [len(canc), canc["InvoiceNo"].nunique(), canc["CustomerID"].nunique(), canc["line_revenue"].sum(),
                        100.0, -canc["line_revenue"].sum() / gross * 100]
    assert abs(canc["line_revenue"].sum() - (rec["Exclude: Matched product cancellations"] + rec["Exclude: Unmatched product cancellations"])) < 0.01
    two = canc[rev_ext & (canc["Quantity"].abs() >= 70000)]["line_revenue"].sum()

    # products driving preventable-candidate cancellations (classes C1, C2)
    prev = canc[canc["cancel_class"].str.startswith("C")]
    prod_sales = nm[nm["line_type"].eq("sale")].groupby("code_u").agg(gross=("line_revenue", "sum"),
                                                                     description=("Description_resolved", "first"))
    pc = prev.groupby("code_u")["line_revenue"].sum().rename("preventable_candidate_value").to_frame().join(prod_sales)
    pc["cancel_rate_pct"] = -pc["preventable_candidate_value"] / pc["gross"] * 100
    pc = pc.sort_values("preventable_candidate_value").head(15).reset_index()
    top10_prod_share = -pc.head(10)["preventable_candidate_value"].sum() / -prev["line_revenue"].sum() * 100
    # customers driving them
    cu = prev.groupby("CustomerID")["line_revenue"].sum().sort_values()
    cust_conc = pd.DataFrame([
        ("Customers with a C1/C2 cancellation", int(cu.size)),
        ("Value, all C1/C2 (£)", round(cu.sum(), 2)),
        ("Share from top 10 customers (%)", round(cu.head(10).sum() / cu.sum() * 100, 2)),
        ("Share from top 10 products (%)", round(top10_prod_share, 2)),
    ], columns=["item", "value"])
    rfm = pd.read_csv("rfm/customer_rfm.csv").set_index("CustomerID")
    prev_seg = prev.assign(CustomerID=prev["CustomerID"].astype("int64")).join(rfm["segment"], on="CustomerID") \
        .groupby("segment")["line_revenue"].agg(["size", "sum"]).rename(columns={"size": "lines", "sum": "value"})
    prev_seg["pct"] = prev_seg["value"] / prev_seg["value"].sum() * 100

    # ---------------- 3. market evidence ----------------
    rs = rfm[rfm["segment"] != "Cancellation-only (not scored)"]
    mk = ["Germany", "France", "Netherlands", "EIRE", "Australia"]
    mix = pd.crosstab(rs["country"], rs["segment"]).reindex(mk)
    netmix = rs.pivot_table(index="country", columns="segment", values="net_revenue", aggfunc="sum").reindex(mk)
    mkt = ctry.loc[mk, ["net_revenue", "orders", "customers", "aov_median", "orders_per_customer",
                         "cancellation_rate_pct", "top_customer_share_pct"]].copy()
    mkt["champions"] = mix["Champions"]
    mkt["at_risk_high_value"] = mix.get("At-Risk High-Value")
    mkt["potential_loyalists"] = mix.get("Potential Loyalists")
    mkt["net_from_at_risk"] = netmix.get("At-Risk High-Value")
    # customer base growth: first-purchase quarter for DE/FR
    fp = rs[rs["country"].isin(["Germany", "France"])].copy()
    fp["first_q"] = pd.to_datetime(fp["first_purchase"]).dt.to_period("Q").astype(str)
    newq = pd.crosstab(fp["first_q"], fp["country"])
    # monthly active customers DE/FR
    act = nm[nm["line_type"].eq("sale") & nm["Country"].isin(["Germany", "France"])]
    actm = act.groupby([act["InvoiceDate"].dt.to_period("M").astype(str), "Country"])["CustomerID"].nunique().unstack()

    assert pd.util.hash_pandas_object(df, index=True).sum() == h0
    outs = {"verification": verif, "cancellation_decomposition": dec.reset_index(), "cancellation_lag_hours": lag_summary,
            "preventable_candidate_products": pc, "preventable_candidate_concentration": cust_conc,
            "preventable_candidate_by_segment": prev_seg.reset_index(), "market_evidence": mkt.reset_index(),
            "de_fr_new_customers_by_quarter": newq.reset_index(), "de_fr_active_customers_by_month": actm.reset_index()}
    for k, t in outs.items():
        t.to_csv(a.out / f"{k}.csv", index=False)
    pd.set_option("display.width", 260, "display.max_colwidth", 80)
    for k, t in outs.items():
        print(f"\n=== {k} ===")
        print(t.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nTwo giant reversed orders (cancel legs): £{two:,.2f} = {two / canc['line_revenue'].sum() * 100:.1f}% of cancelled value")
    print(f"Gross merchandise sales (net view): £{gross:,.2f}")


if __name__ == "__main__":
    main()
