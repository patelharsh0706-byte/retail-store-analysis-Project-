#!/usr/bin/env python3
"""
IE5005 Part 5 — RFM customer segmentation (read-only on retail_prepared.csv.gz).

ELIGIBILITY (all filters on flags written by retail_prepare.py; nothing modified)
  Customer universe   in_customer_analysis  = CustomerID present & in_net_merchandise
                      (excludes non-product charges, stock adjustments, bad-debt
                      entries, corrected double entries; dup_removed rows = 0)
  Purchases (R, F)    universe & line_type == 'sale'      (free items are not purchases)
  Monetary (M)        universe, all lines = sales + free items + product
                      cancellations (matched AND unmatched — every cancellation
                      carries this customer's ID, so it is attributable to them)
  Cancellation-only   customers in the universe with no valid purchase: reported
                      separately, not scored (no recency/frequency defined)

METRICS  reference date = last invoice date in the file + 1 day (2011-12-10)
  Recency   = days from the customer's last valid purchase date to the reference date
  Frequency = distinct valid purchase invoices
  Monetary  = net merchandise revenue (sales − attributable cancellations)

SCORING (1–5, higher is better; equal values always get the same score)
  R  quintile cut points of Recency (lower recency = higher score)
  F  fixed order-count bands chosen at the value boundaries closest to the
     20/40/60/80th percentiles, because 34.7% of customers have exactly 1 order:
     1 → 1, 2 → 2, 3 → 3, 4–5 → 4, 6+ → 5
  M  quintile cut points of Monetary; Monetary ≤ 0 is forced to 1
  Cut points are computed with numpy's default (linear) percentile and applied
  as "score k if value ≤ cut k" (upper bound inclusive), so ties never split.

SEGMENTS (first matching rule wins; every scored customer gets exactly one)
  1 Champions                 R ≥ 4 and F ≥ 4 and M ≥ 4
  2 Loyal Customers           R ≥ 3 and F ≥ 4
  3 At-Risk High-Value        R ≤ 2 and (F ≥ 4 or M ≥ 4)
  4 Potential Loyalists       R ≥ 4 and F in {2, 3}
  5 New or Occasional         R ≥ 3 and F ≤ 3
  6 Low-Engagement            everything else (R ≤ 2, F ≤ 3, M ≤ 3)

Usage:  python retail_rfm.py prepared/retail_prepared.csv.gz --out rfm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retail_eda import verify

SEG_ORDER = ["Champions", "Loyal Customers", "At-Risk High-Value", "Potential Loyalists",
             "New or Occasional", "Low-Engagement"]
F_BANDS = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 5, 4), (6, np.inf, 5)]   # (min, max, score)
LEFT_CENSOR_DAYS = 30        # first purchase within 30 days of data start: earlier history unknown


# ------------------------------------------------------------------ core
def build_customers(df: pd.DataFrame, ref_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = df[df["in_customer_analysis"]]
    buy = u[u["line_type"].eq("sale")]
    canc = u[u["line_type"].eq("cancellation")]
    g = u.groupby("CustomerID")
    cust = pd.DataFrame({
        "net_revenue": g["line_revenue"].sum(),
        "gross_sales": buy.groupby("CustomerID")["line_revenue"].sum(),
        "matched_cancellations": canc[canc["cancellation_status"].eq("matched")].groupby("CustomerID")["line_revenue"].sum(),
        "unmatched_cancellations": canc[canc["cancellation_status"].eq("unmatched")].groupby("CustomerID")["line_revenue"].sum(),
        "frequency": buy.groupby("CustomerID")["InvoiceNo"].nunique(),
        "first_purchase": buy.groupby("CustomerID")["InvoiceDate"].min(),
        "last_purchase": buy.groupby("CustomerID")["InvoiceDate"].max(),
        "cancellation_invoices": canc.groupby("CustomerID")["InvoiceNo"].nunique(),
        "country": g["Country"].agg(lambda s: s.mode().iloc[0]),
        "n_countries": g["Country"].nunique(),
        "has_extreme_qty_line": g["qty_extreme"].any(),
        "has_exact_duplicate_line": g["is_exact_duplicate"].any(),
        "has_pattern_b_repeat": g["repeat_needs_review"].any(),
    })
    for c in ["gross_sales", "matched_cancellations", "unmatched_cancellations", "net_revenue"]:
        cust[c] = cust[c].fillna(0.0).round(2)      # pence: removes float noise such as 1e-12
    cust["frequency"] = cust["frequency"].fillna(0).astype(int)
    cust["cancellation_invoices"] = cust["cancellation_invoices"].fillna(0).astype(int)
    cust["cancellation_only"] = cust["frequency"].eq(0)
    cust["recency_days"] = (ref_date - cust["last_purchase"].dt.normalize()).dt.days
    cust["tenure_days"] = (cust["last_purchase"].dt.normalize() - cust["first_purchase"].dt.normalize()).dt.days
    data_start = df["InvoiceDate"].min().normalize()
    cust["left_censored"] = (cust["first_purchase"].dt.normalize() - data_start).dt.days < LEFT_CENSOR_DAYS
    cust["cancel_rate_value"] = -(cust["matched_cancellations"] + cust["unmatched_cancellations"]) / cust["gross_sales"].replace(0, np.nan)
    cust["net_revenue_excl_unmatched"] = cust["net_revenue"] - cust["unmatched_cancellations"]
    cust.index = cust.index.astype("int64")
    return cust, u


def score(cust: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = cust[~cust["cancellation_only"]].copy()
    r_cut = np.percentile(s["recency_days"], [20, 40, 60, 80])
    m_cut = np.percentile(s["net_revenue"], [20, 40, 60, 80])
    # R: lower recency is better -> recency <= cut1 gives 5
    s["R"] = 5 - np.searchsorted(r_cut, s["recency_days"].to_numpy(), side="left")
    s["F"] = 0
    for lo, hi, sc in F_BANDS:
        s.loc[s["frequency"].between(lo, hi), "F"] = sc
    s["M"] = 1 + np.searchsorted(m_cut, s["net_revenue"].to_numpy(), side="left")
    s.loc[s["net_revenue"] <= 0, "M"] = 1
    assert s[["R", "F", "M"]].isin(range(1, 6)).all().all()
    s["RFM"] = s["R"].astype(str) + s["F"].astype(str) + s["M"].astype(str)
    R, F, M = s["R"], s["F"], s["M"]
    rules = [
        ("Champions", (R >= 4) & (F >= 4) & (M >= 4)),
        ("Loyal Customers", (R >= 3) & (F >= 4)),
        ("At-Risk High-Value", (R <= 2) & ((F >= 4) | (M >= 4))),
        ("Potential Loyalists", (R >= 4) & F.isin([2, 3])),
        ("New or Occasional", (R >= 3) & (F <= 3)),
        ("Low-Engagement", pd.Series(True, index=s.index)),
    ]
    s["segment"] = np.select([m for _, m in rules], [n for n, _ in rules], "UNASSIGNED")
    assert (s["segment"] != "UNASSIGNED").all()
    # ranges for documentation
    def band(series, score_col, k):
        v = series[s[score_col] == k]
        return (f"{v.min():,.0f} – {v.max():,.0f}" if float(v.min()).is_integer() else f"{v.min():,.2f} – {v.max():,.2f}") if len(v) else "none"
    thr = pd.DataFrame([
        {"score": k,
         "recency_days_rule": (f"≤ {r_cut[0]:g}" if k == 5 else f"> {r_cut[-1]:g}" if k == 1
                               else f"> {r_cut[4-k]:g} and ≤ {r_cut[5-k]:g}"),
         "recency_days_observed": band(s["recency_days"], "R", k),
         "frequency_rule": {1: "1 order", 2: "2 orders", 3: "3 orders", 4: "4–5 orders", 5: "6+ orders"}[k],
         "monetary_rule": (f"≤ £{m_cut[0]:,.2f} (incl. all ≤ 0)" if k == 1 else f"> £{m_cut[-1]:,.2f}" if k == 5
                           else f"> £{m_cut[k-2]:,.2f} and ≤ £{m_cut[k-1]:,.2f}"),
         "customers_R": int((s["R"] == k).sum()), "customers_F": int((s["F"] == k).sum()),
         "customers_M": int((s["M"] == k).sum())} for k in [5, 4, 3, 2, 1]])
    return s, thr


def segment_summary(s: pd.DataFrame, total_net: float) -> pd.DataFrame:
    g = s.groupby("segment")
    t = pd.DataFrame({
        "customers": g.size(),
        "gross_sales": g["gross_sales"].sum(),
        "cancellations": g["matched_cancellations"].sum() + g["unmatched_cancellations"].sum(),
        "net_revenue": g["net_revenue"].sum(),
        "median_frequency": g["frequency"].median(),
        "median_recency_days": g["recency_days"].median(),
        "mean_net_per_customer": g["net_revenue"].mean(),
        "median_net_per_customer": g["net_revenue"].median(),
        "customers_net_le_0": g["net_revenue"].apply(lambda x: int((x <= 0).sum())),
        "pct_left_censored": g["left_censored"].mean() * 100,
    }).reindex(SEG_ORDER)
    t.insert(1, "pct_customers", t["customers"] / t["customers"].sum() * 100)
    t.insert(5, "pct_of_attributable_net", t["net_revenue"] / total_net * 100)
    t["cancellation_rate_pct"] = -t["cancellations"] / t["gross_sales"] * 100
    return t


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--out", type=Path, default=Path("rfm"))
    a = ap.parse_args(argv)
    df = pd.read_csv(a.prepared, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"], low_memory=False)
    h0, n0 = pd.util.hash_pandas_object(df, index=True).sum(), len(df)
    checks = verify(df, a.prepared.parent / "revenue_reconciliation.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    ref_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # ---------------- Step 1: eligibility + reconciliation ----------------
    lt = df["line_type"]
    elig = pd.DataFrame([
        ("All rows in prepared file", len(df), df["line_revenue"].sum()),
        ("Excluded: missing CustomerID", int(df["CustomerID"].isna().sum()), df.loc[df["CustomerID"].isna(), "line_revenue"].sum()),
        ("Excluded (ID present): non-product charges", int((df["CustomerID"].notna() & lt.eq("non_product_charge")).sum()),
         df.loc[df["CustomerID"].notna() & lt.eq("non_product_charge"), "line_revenue"].sum()),
        ("Excluded (ID present): non-product cancellations",
         int((df["CustomerID"].notna() & lt.eq("cancellation") & ~df["is_product"]).sum()),
         df.loc[df["CustomerID"].notna() & lt.eq("cancellation") & ~df["is_product"], "line_revenue"].sum()),
        ("Excluded (ID present): stock adjustments", int((df["CustomerID"].notna() & lt.eq("stock_adjustment")).sum()), 0.0),
        ("Excluded (ID present): bad-debt adjustments", int((df["CustomerID"].notna() & lt.eq("accounting_adjustment")).sum()),
         df.loc[df["CustomerID"].notna() & lt.eq("accounting_adjustment"), "line_revenue"].sum()),
        ("Excluded (ID present): corrected double entries (copy + its cancellation)",
         int((df["CustomerID"].notna() & df["corrected_double_entry"] & df["is_product"]).sum()),
         df.loc[df["CustomerID"].notna() & df["corrected_double_entry"] & df["is_product"], "line_revenue"].sum()),
        ("Excluded: confirmed duplicates (dup_removed)", int(df["dup_removed"].sum()), df.loc[df["dup_removed"], "line_revenue"].sum()),
        ("= Customer-analysis rows (in_customer_analysis)", int(df["in_customer_analysis"].sum()),
         df.loc[df["in_customer_analysis"], "line_revenue"].sum()),
    ], columns=["step", "rows", "revenue"])
    assert int(elig["rows"].iloc[1:-1].sum()) + int(elig["rows"].iloc[-1]) == len(df), "eligibility rows do not partition the file"
    assert abs(elig["revenue"].iloc[1:-1].sum() + elig["revenue"].iloc[-1] - elig["revenue"].iloc[0]) < 0.01

    cust, u = build_customers(df, ref_date)
    prior = pd.read_csv(a.prepared.parent / "revenue_views.csv").set_index("view")["revenue"]
    prior_cust_net = float(prior["Customer-analysis net revenue"])
    recon = pd.DataFrame([
        ("Customer-analysis net revenue (revenue_views.csv, Part 2)", prior_cust_net),
        ("Sum of Monetary, scored customers", cust.loc[~cust["cancellation_only"], "net_revenue"].sum()),
        ("Sum of net revenue, cancellation-only customers", cust.loc[cust["cancellation_only"], "net_revenue"].sum()),
        ("Total across all customers", cust["net_revenue"].sum()),
    ], columns=["item", "revenue"])
    recon["difference_vs_part2"] = recon["revenue"] - prior_cust_net
    assert abs(cust["net_revenue"].sum() - prior_cust_net) < 0.01, "customer totals do not reconcile"
    # per-customer check against Part 3 customer net (same definition, independent groupby)
    p3 = df[df["in_net_merchandise"] & df["CustomerID"].notna()].groupby("CustomerID")["line_revenue"].sum()
    p3.index = p3.index.astype("int64")
    assert (cust["net_revenue"] - p3.reindex(cust.index)).abs().max() < 0.01

    # ---------------- Step 2–4 ----------------
    s, thr = score(cust)
    total_attr_net = cust["net_revenue"].sum()
    summ = segment_summary(s, total_attr_net)
    # cancellation rate excluding fully reversed extreme orders (both legs), via existing flags
    rev_pair = u["qty_extreme_reason"].isin(["sale_reversed_by_cancellation", "cancellation_reversing_extreme_sale"])
    ul = u[~rev_pair].assign(CustomerID=lambda d: d["CustomerID"].astype("int64"))
    ul = ul.join(s["segment"], on="CustomerID", how="inner")
    gx = ul[ul["line_type"].eq("sale")].groupby("segment")["line_revenue"].sum()
    cx_ = ul[ul["line_type"].eq("cancellation")].groupby("segment")["line_revenue"].sum()
    summ["cancellation_rate_excl_reversed_extreme_pct"] = (-cx_ / gx * 100).reindex(summ.index)
    co = cust[cust["cancellation_only"]]
    co_row = pd.DataFrame({"customers": [len(co)], "net_revenue": [co["net_revenue"].sum()],
                           "pct_of_attributable_net": [co["net_revenue"].sum() / total_attr_net * 100]},
                          index=["Cancellation-only (not scored)"])
    tot = summ[["customers", "gross_sales", "cancellations", "net_revenue"]].sum()
    total_row = pd.DataFrame({"customers": [tot["customers"]], "pct_customers": [100.0], "gross_sales": [tot["gross_sales"]],
                              "cancellations": [tot["cancellations"]], "net_revenue": [tot["net_revenue"]],
                              "pct_of_attributable_net": [tot["net_revenue"] / total_attr_net * 100],
                              "median_frequency": [s["frequency"].median()], "median_recency_days": [s["recency_days"].median()],
                              "mean_net_per_customer": [s["net_revenue"].mean()], "median_net_per_customer": [s["net_revenue"].median()],
                              "customers_net_le_0": [int((s["net_revenue"] <= 0).sum())],
                              "pct_left_censored": [s["left_censored"].mean() * 100],
                              "cancellation_rate_pct": [-tot["cancellations"] / tot["gross_sales"] * 100],
                              "cancellation_rate_excl_reversed_extreme_pct": [-cx_.sum() / gx.sum() * 100]},
                             index=["All scored customers"])
    summary_full = pd.concat([summ, total_row, co_row])

    # special-case customers to explain
    special = pd.DataFrame([
        ("Scored customers", len(s)),
        ("Cancellation-only customers (not scored)", len(co)),
        ("Scored customers with net revenue < 0", int((s["net_revenue"] < 0).sum())),
        ("Scored customers with net revenue = 0", int((s["net_revenue"].abs() < 0.005).sum())),
        ("Scored customers with any unmatched cancellation", int((s["unmatched_cancellations"] < 0).sum())),
        ("Unmatched cancellations, scored customers (£)", round(s["unmatched_cancellations"].sum(), 2)),
        ("Unmatched cancellations, cancellation-only customers (£)", round(co["unmatched_cancellations"].sum(), 2)),
        ("Matched cancellations, cancellation-only customers (£)", round(co["matched_cancellations"].sum(), 2)),
        ("Scored customers first seen in first 30 days (history may start earlier)", int(s["left_censored"].sum())),
        ("Customers with orders in more than one country", int((cust["n_countries"] > 1).sum())),
        ("Reference date", f"{ref_date:%Y-%m-%d}"),
    ], columns=["item", "value"])

    # concentration
    nets = s["net_revenue"].sort_values(ascending=False)
    conc = pd.DataFrame([{"top_share_of_customers": f"top {p}%", "customers": int(np.ceil(len(nets) * p / 100)),
                          "pct_of_scored_net": nets.head(int(np.ceil(len(nets) * p / 100))).sum() / nets.sum() * 100}
                         for p in [1, 5, 10, 20, 50]])
    top_customers = s.sort_values("net_revenue", ascending=False).head(10)[
        ["net_revenue", "frequency", "recency_days", "country", "segment", "RFM"]]

    # ---------------- Step 7: sensitivity ----------------
    # A) exclude unresolved suspected duplicates: exact-duplicate copies + the
    #    later invoice(s) of each pattern-B repeat group
    pb = df[df["repeat_needs_review"]]
    first_inv = pb.sort_values("InvoiceDate").groupby("repeat_group")["InvoiceNo"].first()
    later_b = df["repeat_needs_review"] & ~df["InvoiceNo"].isin(first_inv.values)
    suspected = df["is_exact_duplicate"] | later_b
    df_a = df.copy()
    df_a["in_customer_analysis"] = df_a["in_customer_analysis"] & ~suspected
    cust_a, _ = build_customers(df_a, ref_date)
    s_a, thr_a = score(cust_a)
    summ_a = segment_summary(s_a, cust_a["net_revenue"].sum())
    moved = s[["segment"]].join(s_a[["segment"]], rsuffix="_excl_dups", how="outer")
    trans_a = pd.crosstab(moved["segment"], moved["segment_excl_dups"], dropna=False).reindex(index=SEG_ORDER, columns=SEG_ORDER, fill_value=0)
    # B) Monetary excluding unmatched cancellations (pre-window purchases)
    cust_b = cust.copy()
    cust_b["net_revenue"] = cust_b["net_revenue_excl_unmatched"]
    s_b, _ = score(cust_b)
    summ_b = segment_summary(s_b, cust_b["net_revenue"].sum())
    sens = pd.DataFrame({
        "customers_base": summ["customers"], "customers_excl_suspected_dups": summ_a["customers"],
        "customers_M_excl_unmatched": summ_b["customers"],
        "net_base": summ["net_revenue"], "net_excl_suspected_dups": summ_a["net_revenue"],
        "net_M_excl_unmatched": summ_b["net_revenue"],
        "pct_net_base": summ["pct_of_attributable_net"], "pct_net_excl_suspected_dups": summ_a["pct_of_attributable_net"],
    })
    stab = pd.DataFrame([
        ("Suspected-duplicate rows removed in test A", int((suspected & df["in_customer_analysis"]).sum())),
        ("Revenue on those rows (£)", round(df.loc[suspected & df["in_customer_analysis"], "line_revenue"].sum(), 2)),
        ("Customers affected (have ≥1 such row)", int(df.loc[suspected & df["in_customer_analysis"], "CustomerID"].nunique())),
        ("Customers changing segment, test A", int((moved["segment"] != moved["segment_excl_dups"]).sum())),
        ("Share of scored customers changing segment, test A (%)", round((moved["segment"] != moved["segment_excl_dups"]).mean() * 100, 2)),
        ("Customers changing segment, test B (M excl. unmatched)", int((s["segment"] != s_b["segment"]).sum())),
        ("Share changing segment, test B (%)", round((s["segment"] != s_b["segment"]).mean() * 100, 2)),
    ], columns=["item", "value"])

    # ---------------- outputs ----------------
    assert len(df) == n0 and pd.util.hash_pandas_object(df, index=True).sum() == h0, "prepared data modified"
    cust_out = cust.join(s[["R", "F", "M", "RFM", "segment"]])
    cust_out["segment"] = cust_out["segment"].fillna("Cancellation-only (not scored)")
    cust_out = cust_out.join(s_a[["segment"]].rename(columns={"segment": "segment_excl_suspected_dups"}))
    cust_out.index.name = "CustomerID"
    cust_out.to_csv(a.out / "customer_rfm.csv")
    outs = {"eligibility": elig, "reconciliation": recon, "special_cases": special, "score_thresholds": thr,
            "segment_summary": summary_full.reset_index(names="segment"),
            "concentration": conc, "top10_customers": top_customers.reset_index(),
            "sensitivity_segments": sens.reset_index(names="segment"), "sensitivity_stability": stab,
            "transition_excl_suspected_dups": trans_a.reset_index(names="segment_base"),
            "score_thresholds_excl_suspected_dups": thr_a, "checks": checks}
    for k, t in outs.items():
        t.to_csv(a.out / f"{k}.csv", index=False)

    # ---------------- chart ----------------
    make_chart(summ, a.out / "rfm_segments_customers_vs_revenue")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for k in ["eligibility", "reconciliation", "special_cases", "score_thresholds", "segment_summary",
              "concentration", "top10_customers", "sensitivity_segments", "sensitivity_stability",
              "transition_excl_suspected_dups"]:
        print(f"\n=== {k} ===")
        print(outs[k].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return outs


def make_chart(summ: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "xtick.color": INK2, "ytick.color": INK2})
    t = summ.iloc[::-1]
    y = np.arange(len(t))
    h = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(y + h / 2, t["pct_customers"], height=h, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="Share of scored customers")
    ax.barh(y - h / 2, t["pct_of_attributable_net"], height=h, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
            label="Share of attributable net revenue")
    for yi, (c, r) in enumerate(zip(t["pct_customers"], t["pct_of_attributable_net"])):
        ax.text(c + 0.6, yi + h / 2, f"{c:.1f}%", va="center", fontsize=8.5, color=INK)
        ax.text(r + 0.6, yi - h / 2, f"{r:.1f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}\n({int(n):,} customers)" for s, n in zip(t.index, t["customers"])], fontsize=9.5)
    ax.set_xlabel("Percent (%)", color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(t["pct_customers"].max(), t["pct_of_attributable_net"].max()) * 1.15)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Champions are 21% of customers but two-thirds of net revenue", loc="left",
                 fontsize=13, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.02, "RFM segments, customers with an ID, 1 Dec 2010 – 9 Dec 2011 (net merchandise revenue after attributable cancellations)",
            transform=ax.transAxes, fontsize=9, color=INK2)
    cap = ("Each segment's share of the 4,334 scored customers (blue) and of attributable net revenue (orange); cancellation-only "
           "customers are excluded. Revenue shares can sum to slightly more than 100% because the 29 cancellation-only customers net to a negative amount.")
    fig.text(0.01, 0.012, textwrap.fill(cap, 150), fontsize=8.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
