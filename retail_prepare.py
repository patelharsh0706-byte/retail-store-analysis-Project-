#!/usr/bin/env python3
"""
Online Retail — analysis-ready layer (decisions agreed after the audit).

Original columns are never changed. Every decision is a NEW column, so any
analysis can be reproduced with or without it:

  Decision 1  Missing Description
      kept for revenue validation; excluded from product-name analysis only
      when the product cannot be identified at all.
      -> product_identifiable, Description_resolved
  Decision 2  UnitPrice = 0
      legitimate free items retained; non-sale entries separated.
      -> line_type ('free_item' vs 'stock_adjustment')
  Decision 3  Non-standard StockCodes
      every code classified as product vs a named non-product class.
      -> code_class, is_product
  Decision 4  Description formatting
      whitespace standardised in a new column; original kept.
      -> Description_std
  Quantity outliers: NOT treated here (pending decision).

Usage:  python retail_prepare.py Online_Retail.xlsx --out prepared
Writes: prepared/retail_prepared.csv.gz, prepared/decision_log.md
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- decision 3
# StockCode -> class. Anything matching STANDARD_CODE is a product. The DCGS*
# codes are dotcom-shop products (party bags, dog collars...), so they are
# products too; their rows with no description / 'ebay' notes fall out as
# stock adjustments through the line_type rules, not through the code class.
STANDARD_CODE = r"\d{5}[A-Za-z]{0,2}"
NON_PRODUCT_CODES = {
    "POST": "shipping", "DOT": "shipping", "C2": "shipping",
    "BANK CHARGES": "fee", "AMAZONFEE": "fee", "CRUK": "fee",
    "D": "discount",
    "M": "manual_adjustment",
    "S": "samples",
    "B": "bad_debt_adjustment",
}
PRODUCT_PATTERNS = [r"DCGS\w*", r"PADS"]          # non-standard but merchandise
VOUCHER_PATTERN = r"gift_\d+_\d+"                   # gift voucher: liability, not goods


def classify_code(code: pd.Series) -> pd.Series:
    c = code.astype(str).str.strip()
    cu = c.str.upper()
    out = pd.Series("unclassified", index=code.index, dtype=object)
    out[c.str.fullmatch(STANDARD_CODE)] = "product"
    for pat in PRODUCT_PATTERNS:
        out[cu.str.fullmatch(pat)] = "product"
    out[c.str.fullmatch(VOUCHER_PATTERN, case=False)] = "gift_voucher"
    for k, v in NON_PRODUCT_CODES.items():
        out[cu == k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=Path("prepared"))
    ap.add_argument("--dedupe", choices=["flag", "remove_replayed", "remove_all"], default="flag",
                    help="which exact-duplicate copies to exclude from every in_* filter (default: none)")
    a = ap.parse_args(argv)

    dtype = {"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str}
    raw = pd.read_excel(a.input, dtype=dtype)
    before = pd.util.hash_pandas_object(raw, index=True).sum()
    df = raw.copy()
    df.insert(0, "source_row", np.arange(2, len(df) + 2))

    inv = df["InvoiceNo"].astype(str).str.strip().str.upper()
    code_u = df["StockCode"].astype(str).str.strip().str.upper()
    qty, price, cust = df["Quantity"], df["UnitPrice"], df["CustomerID"]
    df["line_revenue"] = qty * price

    # ------------------------------------------------------------ decision 4
    d = df["Description"]
    df["Description_std"] = d.str.strip().str.replace(r"\s+", " ", regex=True)

    # ------------------------------------------------------------ decision 3
    df["code_class"] = classify_code(df["StockCode"])
    df["is_product"] = df["code_class"].eq("product")

    # ------------------------------------------------------------ decision 1
    # A missing description is still identifiable if the same StockCode carries
    # a description on other rows. Canonical name = most frequent standardised
    # description among priced rows of that code (ties broken alphabetically,
    # so the result is deterministic).
    named = df[df["Description_std"].notna() & (price > 0)]
    canon = (named.assign(_c=code_u[named.index])
             .groupby(["_c", "Description_std"]).size().rename("n").reset_index()
             .sort_values(["_c", "n", "Description_std"], ascending=[True, False, True])
             .drop_duplicates("_c").set_index("_c")["Description_std"])
    df["Description_canonical"] = code_u.map(canon)
    df["Description_resolved"] = df["Description_std"].fillna(df["Description_canonical"])
    df["description_source"] = np.select(
        [df["Description_std"].notna(), df["Description_canonical"].notna()],
        ["original", "borrowed_from_stockcode"], "none")
    df["product_identifiable"] = df["is_product"] & df["Description_resolved"].notna()

    # ------------------------------------------------------------ decision 2
    # line_type is assigned in priority order; first matching rule wins.
    is_cancel = inv.str.startswith("C")
    is_acct = inv.str.startswith("A")
    rules = [
        ("accounting_adjustment", is_acct),
        ("cancellation", is_cancel),
        ("non_product_charge", ~df["is_product"]),
        # price 0 with no customer, or negative qty: warehouse write-off / note
        ("stock_adjustment", (price == 0) & (cust.isna() | (qty <= 0))),
        ("stock_adjustment", (qty < 0)),
        # price 0, real product, named customer, positive qty: given away
        ("free_item", (price == 0) & cust.notna() & (qty > 0)),
        ("sale", (price > 0) & (qty > 0)),
    ]
    df["line_type"] = np.select([m for _, m in rules], [n for n, _ in rules], "other")

    # ------------------------------------------------------------ decision 6
    # Exact duplicates: flag every copy after the first. Removal only when
    # --dedupe remove is passed (pending confirmation of legitimacy).
    orig_cols = list(raw.columns)
    df["is_exact_duplicate"] = df[orig_cols].duplicated(keep="first")
    df["dup_group_size"] = df.groupby(orig_cols, dropna=False)["source_row"].transform("size")
    # Evidence that a copy is a repeated RECORDING (not a repeat purchase):
    #   replayed_run  - copy is part of >=2 consecutive lines with >=2 distinct
    #                   items that repeat an earlier run on the same invoice
    #   consecutive   - copy sits on the row directly after its original
    #   scattered     - copy appears elsewhere on the invoice
    # Rule agreed: remove only with reasonable evidence. Review of the data
    # showed replayed runs are sets of qty-1 items bought twice and 70% of
    # other copies are qty-1 basket adds, so by default NOTHING is removed.
    key = pd.util.hash_pandas_object(df[orig_cols], index=False)
    ev = pd.Series(pd.NA, index=df.index, dtype=object)
    prev_same = key.eq(key.shift(1)) & inv.eq(inv.shift(1))
    ev[df["is_exact_duplicate"]] = np.where(prev_same[df["is_exact_duplicate"]], "consecutive", "scattered")
    for _, g in df[inv.isin(inv[df["is_exact_duplicate"]])].groupby(inv, sort=False):
        ks, cs, ix, n = key[g.index].tolist(), code_u[g.index].tolist(), g.index.tolist(), len(g)
        for j in range(n):
            for i in range(j + 1, n):
                if ks[i] != ks[j]:
                    continue
                L = 0
                while i + L < n and j + L < i and ks[i + L] == ks[j + L]:
                    L += 1
                if L >= 2 and len(set(cs[i:i + L])) >= 2:
                    for t in range(L):
                        if df.at[ix[i + t], "is_exact_duplicate"]:
                            ev[ix[i + t]] = "replayed_run"
    df["dup_evidence"] = ev
    REMOVE_EVIDENCE = {"flag": set(), "remove_replayed": {"replayed_run"},
                       "remove_all": {"replayed_run", "consecutive", "scattered"}}[a.dedupe]
    df["dup_removed"] = df["dup_evidence"].isin(REMOVE_EVIDENCE)

    # Whole invoice recorded twice under a new number: same customer, identical
    # multiset of lines, within 60 minutes. FLAG ONLY (not an exact duplicate).
    sig = code_u + "|" + qty.astype(str) + "|" + price.astype(str)
    ih = (df.assign(_inv=inv, _sig=sig).groupby("_inv")
          .agg(cust=("CustomerID", "first"), t=("InvoiceDate", "min"),
               lines=("_sig", lambda x: "#".join(sorted(x)))).reset_index())
    ih = ih[ih["cust"].notna()].sort_values("t")
    ih["prev_t"] = ih.groupby(["cust", "lines"])["t"].shift(1)
    ih["next_t"] = ih.groupby(["cust", "lines"])["t"].shift(-1)
    hit = ((ih["t"] - ih["prev_t"]) <= pd.Timedelta("60min")) | ((ih["next_t"] - ih["t"]) <= pd.Timedelta("60min"))
    df["possible_repeat_invoice"] = inv.isin(ih.loc[hit, "_inv"])

    # ------------------------------------------------------------ decision 7
    # Cancellations stay in the data; subclassify by whether an earlier
    # purchase of the same StockCode by the same customer exists in the file.
    buy = df[(df["line_type"].isin(["sale", "free_item"])) & cust.notna()]
    first_buy = buy.assign(_c=code_u[buy.index]).groupby(["CustomerID", "_c"])["InvoiceDate"].min()
    cx = df.index[is_cancel]
    fb = first_buy.reindex(pd.MultiIndex.from_arrays([cust[cx], code_u[cx]])).to_numpy()
    matched = pd.notna(fb) & (fb <= df.loc[cx, "InvoiceDate"].to_numpy())
    df["cancellation_match"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[cx, "cancellation_match"] = (np.where(matched, "matched", "unmatched_no_prior_purchase"))
    df.loc[cx[cust[cx].isna().to_numpy()], "cancellation_match"] = "unmatched_no_customer"
    df["cancellation_status"] = df["cancellation_match"].map(
        lambda v: v if pd.isna(v) else ("matched" if v == "matched" else "unmatched"))

    # ------------------------------------------------------------ decision 11
    # Classify every possible repeat invoice group (same customer, identical
    # lines, within 60 min) into one pattern. Rules are applied in this order:
    #   D_manual_correction_net_zero  non-product group with a mirror group
    #                                 (same customer and codes, opposite sign)
    #                                 within 60 min. Net zero; no action.
    #   C_habitual_repeat_purchase    customer has >= 3 identical invoices in the
    #                                 file (a reorder habit). Legitimate; no action.
    #   A_corrected_double_entry      sale group where every line of the later
    #                                 copy is reversed by a cancellation line
    #                                 (same customer, code, -qty, price) within
    #                                 60 min. Keep rows; exclude the extra copy
    #                                 AND its correcting cancellation from
    #                                 gross/net merchandise and customer analysis
    #                                 (net revenue unchanged; gross and order
    #                                 counts no longer inflated).
    #   B_uncorrected_repeat_review   any other sale group. Keep; needs source check.
    #   E_repeat_cancellation_supported / _unverifiable
    #                                 cancellation group: earlier purchases do /
    #                                 do not cover all copies. Keep.
    WINDOW = pd.Timedelta("60min")
    ih_ = ih.loc[hit].copy()
    ih_["grp"] = ih_.groupby(["cust", "lines"]).ngroup()
    ih_["is_c"] = ih_["_inv"].str.startswith("C")
    R_line = df["line_revenue"]
    line_df = df.assign(_inv=inv, _c=code_u)
    lines_of = {k: g for k, g in line_df[line_df["_inv"].isin(ih_["_inv"])].groupby("_inv")}
    cx_lines = line_df[is_cancel & cust.notna()]
    df["repeat_group"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["repeat_pattern"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df["repeat_role"] = pd.Series(pd.NA, index=df.index, dtype=object)
    groups = {gid: g.sort_values("t") for gid, g in ih_.groupby("grp")}
    pattern, used_cx = {}, set()
    def group_lines(g):
        return line_df.index[line_df["_inv"].isin(g["_inv"])]
    # D: group + same-customer, same-code cancellations within +/-60 min net to 0
    for gid, g in groups.items():
        if g["is_c"].iloc[0] or gid in pattern:
            continue
        gl = group_lines(g)
        c0, codes = g["cust"].iloc[0], set(code_u[gl])
        linked = cx_lines[(cx_lines["CustomerID"] == c0) & cx_lines["_c"].isin(codes)
                          & (cx_lines["InvoiceDate"] >= g["t"].min() - WINDOW)
                          & (cx_lines["InvoiceDate"] <= g["t"].max() + WINDOW)]
        if len(linked) and abs(R_line[gl].sum() + R_line[linked.index].sum()) < 0.01:
            pattern[gid] = "D_reversed_net_zero"
            df.loc[linked.index, "repeat_role"] = "net_zero_reversal"
            used_cx.update(linked.index)
            for gid2, g2 in groups.items():          # a linked repeat-cancellation group joins D
                if g2["is_c"].iloc[0] and set(group_lines(g2)) <= set(linked.index):
                    pattern[gid2] = "D_reversed_net_zero"
    # A: every line of the later copy reversed exactly by a cancellation line
    for gid, g in groups.items():
        if g["is_c"].iloc[0] or gid in pattern:
            continue
        c0, copy = g["cust"].iloc[0], lines_of[g["_inv"].iloc[-1]]
        cand = cx_lines[(cx_lines["CustomerID"] == c0) & (cx_lines["InvoiceDate"] >= g["t"].min())
                        & (cx_lines["InvoiceDate"] <= g["t"].max() + WINDOW) & ~cx_lines.index.isin(list(used_cx))]
        picked = []
        for _, ln in copy.iterrows():
            m = cand[(cand["_c"] == ln["_c"]) & (cand["Quantity"] == -ln["Quantity"])
                     & (cand["UnitPrice"] == ln["UnitPrice"]) & ~cand.index.isin(picked)]
            if len(m):
                picked.append(m.index[0])
        if len(picked) == len(copy):
            pattern[gid] = "A_corrected_double_entry"
            used_cx.update(picked)
            df.loc[copy.index, "repeat_role"] = "extra_copy"
            df.loc[picked, ["repeat_role", "repeat_group", "repeat_pattern"]] = ["correcting_cancellation", gid, "A_corrected_double_entry"]
    # C: the repeat itself shows a habit (>= 3 identical invoices in one group,
    # or the customer has such a group); otherwise B
    habit_cust = {g["cust"].iloc[0] for g in groups.values() if len(g) >= 3 and not g["is_c"].iloc[0]}
    for gid, g in groups.items():
        if gid in pattern:
            continue
        c0 = g["cust"].iloc[0]
        if g["is_c"].iloc[0]:
            l = lines_of[g["_inv"].iloc[0]]
            need = (-l.groupby("_c")["Quantity"].sum()) * len(g)
            prior = line_df[(cust == c0) & line_df["_c"].isin(need.index) & line_df["line_type"].isin(["sale", "free_item"])
                            & (df["InvoiceDate"] < g["t"].min())].groupby("_c")["Quantity"].sum()
            ok = (prior.reindex(need.index).fillna(0) >= need).all()
            pattern[gid] = "E_repeat_cancellation_supported" if ok else "E_repeat_cancellation_unverifiable"
        elif c0 in habit_cust:
            pattern[gid] = "C_habitual_repeat_purchase"
        else:
            pattern[gid] = "B_uncorrected_repeat_review"
    for gid, g in groups.items():
        idx = line_df.index[line_df["_inv"].isin(g["_inv"])]
        df.loc[idx, "repeat_group"] = gid
        df.loc[idx, "repeat_pattern"] = pattern[gid]
        df.loc[idx[df.loc[idx, "repeat_role"].isna()], "repeat_role"] = "member"
    df["repeat_needs_review"] = df["repeat_pattern"].eq("B_uncorrected_repeat_review")
    df["corrected_double_entry"] = df["repeat_role"].isin(["extra_copy", "correcting_cancellation"])

    # ------------------------------------------------------------ decision 8
    # Negative UnitPrice: all are 'B' Adjust bad debt on 'A' invoices -> the
    # accounting_adjustment line_type above. Explicit flag for auditability.
    df["negative_price_adjustment"] = price < 0

    # ------------------------------------------------------------ decision 10
    # Extreme quantity: FLAG + reason only. No row is excluded on this basis.
    sale = df["line_type"].eq("sale")
    med = qty.where(sale).groupby(code_u).transform("median")
    n_sale = sale.groupby(code_u).transform("sum")
    rel = sale & (n_sale >= 20) & (qty >= 100) & (qty >= 50 * med)
    df["qty_extreme"] = (qty.abs() >= 1000) | rel
    df["qty_vs_product_median"] = (qty / med).round(1).where(df["qty_extreme"])
    # pair an extreme positive line with an identical-size cancellation
    # (same customer, code, -qty, at or after the sale) and mark both rows
    pos = df[df["qty_extreme"] & (qty > 0) & ~is_cancel & cust.notna()]
    neg = df[is_cancel & cust.notna()]
    pairs = (pos.reset_index().assign(_c=code_u[pos.index].values)
             .merge(neg.reset_index().assign(_c=code_u[neg.index].values, _q=-neg["Quantity"].values),
                    left_on=["CustomerID", "_c", "Quantity"], right_on=["CustomerID", "_c", "_q"], suffixes=("", "_cx")))
    pairs = pairs[pairs["InvoiceDate_cx"] >= pairs["InvoiceDate"]].sort_values("InvoiceDate_cx") \
        .drop_duplicates("index").drop_duplicates("index_cx")
    reversed_sale = df.index.isin(pairs["index"])
    reversing_cx = df.index.isin(pairs["index_cx"])
    df["qty_extreme_reason"] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[df["qty_extreme"] | reversing_cx, "qty_extreme_reason"] = np.select(
        [reversed_sale, reversing_cx, df["line_type"].eq("stock_adjustment"), df["line_type"].eq("free_item"),
         is_cancel, df["line_type"].eq("non_product_charge")],
        ["sale_reversed_by_cancellation", "cancellation_reversing_extreme_sale", "stock_adjustment",
         "free_item_bulk", "large_cancellation", "non_product"],
        "bulk_order_not_reversed")[(df["qty_extreme"] | reversing_cx).to_numpy()]
    df.loc[reversing_cx, "qty_extreme"] = True

    # --------------------------------------------------- convenience filters
    df["in_revenue_validation"] = True              # decision 1: every row kept
    df["in_merchandise_sales"] = df["line_type"].isin(["sale", "free_item"]) & ~df["negative_price_adjustment"]
    df["in_product_name_analysis"] = df["product_identifiable"] & df["line_type"].isin(["sale", "free_item"])
    # decision 5: valid transactions kept, but customer analysis needs an ID
    df["is_product_cancellation"] = df["line_type"].eq("cancellation") & df["is_product"]
    df["in_net_merchandise"] = df["in_merchandise_sales"] | df["is_product_cancellation"]
    df["in_customer_analysis"] = cust.notna() & df["in_net_merchandise"]
    for col in ["in_revenue_validation", "in_merchandise_sales", "in_net_merchandise",
                "in_product_name_analysis", "in_customer_analysis"]:
        df.loc[df["dup_removed"], col] = False
    # decision 11 (pattern A): extra copy + its correcting cancellation leave
    # every merchandise / customer view; they stay in revenue validation.
    for col in ["in_merchandise_sales", "in_net_merchandise", "in_product_name_analysis", "in_customer_analysis"]:
        df.loc[df["corrected_double_entry"], col] = False

    # ------------------------------------------------- revenue reconciliation
    R = df["line_revenue"]
    keep = ~df["dup_removed"]
    def rs(mask):
        mask = mask & keep
        return int(mask.sum()), round(float(R[mask].sum()), 2)
    lt_ = df["line_type"]
    steps = [
        ("Total, all rows as loaded", None, (len(df), round(float(R.sum()), 2)), "start"),
        ("Exact duplicates removed", None, (int(df["dup_removed"].sum()), round(float(R[df["dup_removed"]].sum()), 2)), "less"),
        ("Accounting adjustments (bad debt)", lt_.eq("accounting_adjustment"), None, "less"),
        ("Non-product charges (postage, fees, manual...)", lt_.eq("non_product_charge"), None, "less"),
        ("Non-product cancellations", lt_.eq("cancellation") & ~df["is_product"], None, "less"),
        ("Stock adjustments (write-offs, notes)", lt_.eq("stock_adjustment"), None, "less"),
        ("Corrected double entries: extra copies (product lines)",
         df["repeat_role"].eq("extra_copy") & df["is_product"] & lt_.isin(["sale", "free_item"]), None, "less"),
        ("Corrected double entries: their correcting cancellations (product lines)",
         df["repeat_role"].eq("correcting_cancellation") & df["is_product_cancellation"], None, "less"),
        ("= Net merchandise revenue", df["in_net_merchandise"], None, "subtotal"),
        ("Matched product cancellations", df["in_net_merchandise"] & df["is_product_cancellation"] & df["cancellation_status"].eq("matched"), None, "less"),
        ("Unmatched product cancellations", df["in_net_merchandise"] & df["is_product_cancellation"] & df["cancellation_status"].eq("unmatched"), None, "less"),
        ("= Gross merchandise sales", df["in_merchandise_sales"], None, "subtotal"),
    ]
    rec, run_rows, run_rev = [], 0, 0.0
    for label, mask, fixed, kind in steps:
        rows_, rev_ = fixed if fixed else rs(mask)
        if kind == "start":
            run_rows, run_rev = rows_, rev_
        elif kind == "less":
            run_rows, run_rev = run_rows - rows_, round(run_rev - rev_, 2)
        else:
            assert rows_ == run_rows and abs(rev_ - run_rev) < 0.01, (label, rows_, run_rows, rev_, run_rev)
        rec.append({"step": ("Exclude: " if kind == "less" else "") + label,
                    "category_rows": rows_, "category_revenue": rev_,
                    "effect_on_total": round(-rev_, 2) if kind == "less" else None,
                    "running_rows": run_rows, "running_revenue": run_rev})
    recon = pd.DataFrame(rec)
    # customer view and sensitivities (not part of the waterfall)
    no_cust = df["in_net_merchandise"] & cust.isna()
    sens = pd.DataFrame([
        {"view": "Net merchandise revenue", **dict(zip(["rows", "revenue"], rs(df["in_net_merchandise"])))},
        {"view": "  of which rows without CustomerID", **dict(zip(["rows", "revenue"], rs(no_cust)))},
        {"view": "Customer-analysis net revenue", **dict(zip(["rows", "revenue"], rs(df["in_customer_analysis"])))},
        {"view": "Sensitivity: exact duplicates in net merchandise (kept)",
         **dict(zip(["rows", "revenue"], rs(df["in_net_merchandise"] & df["is_exact_duplicate"])))},
        {"view": "Sensitivity: extreme sales + their reversing cancellations",
         **dict(zip(["rows", "revenue"], rs(df["in_net_merchandise"] & df["qty_extreme_reason"].isin(
             ["sale_reversed_by_cancellation", "cancellation_reversing_extreme_sale"]))))},
        {"view": "Sensitivity: pattern B repeat invoices awaiting review (kept)",
         **dict(zip(["rows", "revenue"], rs(df["in_net_merchandise"] & df["repeat_needs_review"])))},
    ])
    # confirmation: every row in exactly one line_type, totals tie out
    lt_tot = df.groupby("line_type")["line_revenue"].agg(["size", "sum"])
    assert lt_tot["size"].sum() == len(df) and abs(lt_tot["sum"].sum() - R.sum()) < 0.01
    conf = pd.DataFrame([
        {"check": "Rows loaded", "value": f"{len(df):,}"},
        {"check": "Rows across line types (must equal rows loaded)", "value": f"{int(lt_tot['size'].sum()):,}"},
        {"check": "Revenue, all rows", "value": f"{R.sum():,.2f}"},
        {"check": "Revenue across line types (must equal above)", "value": f"{lt_tot['sum'].sum():,.2f}"},
        {"check": "Rows removed as duplicates", "value": f"{int(df['dup_removed'].sum()):,}"},
        {"check": "Rows eligible for customer analysis", "value": f"{int(df['in_customer_analysis'].sum()):,}"},
        {"check": "Distinct customers eligible", "value": f"{df.loc[df['in_customer_analysis'], 'CustomerID'].nunique():,}"},
        {"check": "Distinct CustomerIDs in raw file", "value": f"{cust.nunique():,}"},
        {"check": "Customer-eligible rows with missing CustomerID (must be 0)",
         "value": f"{int((df['in_customer_analysis'] & cust.isna()).sum()):,}"},
        {"check": "Customer-eligible rows that are non-product (must be 0)",
         "value": f"{int((df['in_customer_analysis'] & ~df['is_product']).sum()):,}"},
    ])
    ctx_recon = (recon, sens, conf)

    after = pd.util.hash_pandas_object(raw, index=True).sum()
    assert before == after, "raw data was modified"
    for col in raw.columns:                          # originals untouched in output too
        assert df[col].equals(raw[col]), col

    a.out.mkdir(parents=True, exist_ok=True)
    recon, sens, conf = ctx_recon
    recon.to_csv(a.out / "revenue_reconciliation.csv", index=False)
    sens.to_csv(a.out / "revenue_views.csv", index=False)
    df.to_csv(a.out / "retail_prepared.csv.gz", index=False, compression="gzip")

    # ------------------------------------------------------------ decision log
    def t(x):
        return x.to_markdown()
    gross = df["line_revenue"].sum()
    lt = df.groupby("line_type").agg(rows=("line_revenue", "size"), revenue=("line_revenue", "sum"))
    lt["pct_revenue"] = (lt["revenue"] / gross * 100).round(2)
    cc = df.groupby("code_class").agg(rows=("line_revenue", "size"), revenue=("line_revenue", "sum"),
                                      codes=("StockCode", lambda s: s.str.strip().str.upper().nunique()))
    miss = df[df["Description"].isna()]
    log = [
        "# Decision log — retail_prepared.csv.gz\n",
        f"Input sha256 `{hashlib.sha256(a.input.read_bytes()).hexdigest()[:16]}…` · rows {len(df):,} · "
        f"net revenue (all rows) {gross:,.2f}. Original 8 columns are unchanged (asserted).\n",
        "## Rows by line_type\n", t(lt.round(2)), "\n",
        "## Rows by code_class\n", t(cc.round(2)), "\n",
        "## Decision 1 — missing Description\n",
        f"{len(miss):,} rows; revenue on them = {miss['line_revenue'].sum():,.2f}. "
        f"Description recoverable from StockCode for {(miss['description_source']=='borrowed_from_stockcode').sum():,}; "
        f"not identifiable: {(miss['description_source']=='none').sum():,}.\n",
        "## Decision 2 — UnitPrice = 0\n",
        t(df[price == 0].groupby("line_type").size().rename("rows").to_frame()), "\n",
        "Free items (review list):\n",
        df.loc[df["line_type"] == "free_item", ["InvoiceNo", "StockCode", "Description_std", "Quantity", "CustomerID", "InvoiceDate"]]
          .to_markdown(index=False), "\n",
        "## Decision 5 — missing CustomerID\n",
        f"{cust.isna().sum():,} rows kept; excluded from in_customer_analysis. "
        f"Revenue in customer analysis {df.loc[df['in_customer_analysis'], 'line_revenue'].sum():,.2f} "
        f"vs merchandise sales {df.loc[df['in_merchandise_sales'], 'line_revenue'].sum():,.2f}.\n",
        "## Confirmation checks\n", conf.to_markdown(index=False, disable_numparse=True), "\n",
        "## Revenue reconciliation\n", recon.to_markdown(index=False, floatfmt=",.2f"), "\n",
        sens.to_markdown(index=False, floatfmt=",.2f"), "\n",
        "## Decision 6 — exact duplicates\n",
        f"Mode: **{a.dedupe}** (removed: {int(df['dup_removed'].sum()):,}). Copies after the first: "
        f"{df['is_exact_duplicate'].sum():,}; revenue on them {df.loc[df['is_exact_duplicate'], 'line_revenue'].sum():,.2f}.\n",
        t(df[df["is_exact_duplicate"]].groupby("dup_evidence").agg(
            rows=("line_revenue", "size"), revenue=("line_revenue", "sum"),
            share_qty_1=("Quantity", lambda q: round((q == 1).mean(), 3)))), "\n",
        "## Decision 11 — whole invoices repeated (same customer, identical lines, within 60 min)\n",
        t(df[df["possible_repeat_invoice"]].groupby("repeat_pattern").agg(
            groups=("repeat_group", "nunique"), invoices=("InvoiceNo", "nunique"), rows=("line_revenue", "size"),
            revenue=("line_revenue", "sum")).round(2)), "\n",
        "Rows excluded from merchandise and customer views as corrected double entries:\n",
        t(df[df["corrected_double_entry"]].groupby("repeat_role").agg(
            rows=("line_revenue", "size"), invoices=("InvoiceNo", "nunique"), revenue=("line_revenue", "sum")).round(2)), "\n",
        "Groups flagged for source-system review (pattern B):\n",
        df.loc[df["repeat_needs_review"]].groupby(["repeat_group", "InvoiceNo"]).agg(
            customer=("CustomerID", "first"), time=("InvoiceDate", "min"), lines=("line_revenue", "size"),
            revenue=("line_revenue", "sum")).reset_index().to_markdown(index=False), "\n",
        "## Decision 7 — cancellations\n",
        t(df.groupby("cancellation_match").agg(rows=("line_revenue", "size"), revenue=("line_revenue", "sum")).round(2)), "\n",
        "## Decision 8 — negative UnitPrice\n",
        df.loc[df["StockCode"].astype(str).str.strip().str.upper().eq("B"),
               ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "line_type", "in_merchandise_sales"]]
          .to_markdown(index=False), "\n",
        "## Decision 10 — extreme quantities (flagged, not excluded)\n",
        t(df[df["qty_extreme"]].groupby("qty_extreme_reason").agg(
            rows=("line_revenue", "size"), units=("Quantity", "sum"), revenue=("line_revenue", "sum")).round(2)), "\n",
        df.loc[df["qty_extreme"], ["InvoiceNo", "StockCode", "Description_std", "Quantity", "UnitPrice", "CustomerID",
                                   "InvoiceDate", "qty_vs_product_median", "qty_extreme_reason"]]
          .sort_values("Quantity", key=abs, ascending=False).head(40).to_markdown(index=False), "\n",
        "## Decision 4 — whitespace\n",
        f"{(df['Description'].notna() & (df['Description'] != df['Description_std'])).sum():,} rows changed in Description_std; "
        f"distinct descriptions {df['Description'].nunique():,} -> {df['Description_std'].nunique():,}.\n",
    ]
    (a.out / "decision_log.md").write_text("\n".join(log), encoding="utf-8")
    print(lt.round(2).to_string()); print(); print(cc.round(2).to_string())


if __name__ == "__main__":
    main()
