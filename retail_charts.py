#!/usr/bin/env python3
"""
IE5005 Part 4 — four publication charts from retail_prepared.csv.gz.

Same subset definitions as Part 3 (retail_eda.py / retail_eda_part3b.py):
  VIEW 1  valid merchandise sales  in_merchandise_sales & line_type == 'sale'
  VIEW 2  net merchandise          in_net_merchandise
No record is modified or removed. Skewed variables are shown on log scales;
outliers stay in every chart and the largest ones are labelled.

Usage:  python retail_charts.py prepared/retail_prepared.csv.gz --out charts
Writes: one PNG (300 dpi) and one SVG per chart, chart_data_*.csv, captions.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from scipy import stats

from retail_eda import verify

# ---- palette (validated: blue/orange pass all CVD + contrast checks on light surface)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_LIGHT = "#86b6ef"
UK = "United Kingdom"
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlepad": 26,
})


def gbp(x, _=None):
    if abs(x) >= 1e6:
        return f"£{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"£{x/1e3:,.0f}k"
    return f"£{x:,.0f}" if x >= 1 else f"£{x:.2g}"


def finish(fig, ax, title, subtitle, caption, path: Path):
    ax.set_title(title, color=INK)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    import textwrap
    width = int(fig.get_figwidth() * 15.5)
    wrapped = textwrap.fill(caption, width)
    fig.text(0.01, 0.012, wrapped, fontsize=8.5, color=INK2, va="bottom")
    n_lines = wrapped.count("\n") + 1
    fig.tight_layout(rect=(0, 0.03 + 0.032 * n_lines, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=DPI)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--out", type=Path, default=Path("charts"))
    a = ap.parse_args(argv)
    df = pd.read_csv(a.prepared, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"], low_memory=False)
    h0 = pd.util.hash_pandas_object(df, index=True).sum()
    verify(df, a.prepared.parent / "revenue_reconciliation.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    v1 = df[df["in_merchandise_sales"] & df["line_type"].eq("sale")]
    v2 = df[df["in_net_merchandise"]]
    caps = []

    # ================= Chart 1: order-value histogram (log x) =================
    orders = v1.groupby("InvoiceNo")["line_revenue"].sum()
    med, mean = orders.median(), orders.mean()
    n_over_10k = int((orders > 10_000).sum())
    bins = np.logspace(np.log10(orders.min()), np.log10(orders.max()), 60)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.hist(orders, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mt.FuncFormatter(gbp))
    ax.set_xlabel("Order value (£, log scale) — one order = one invoice of valid merchandise sales")
    ax.set_ylabel("Number of orders")
    ax.yaxis.set_major_formatter(mt.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    ytop = ax.get_ylim()[1]
    for x, lab, col, ls in [(med, f"Median {gbp(med)}", INK, "-"), (mean, f"Mean {gbp(mean)}", ORANGE, "--")]:
        ax.axvline(x, color=col, lw=1.6, ls=ls)
    ax.text(med / 1.08, ytop * 0.95, f"Median £{med:,.0f}", ha="right", va="top", color=INK, fontsize=9.5, fontweight="bold")
    ax.text(mean * 1.08, ytop * 0.95, f"Mean £{mean:,.0f}", ha="left", va="top", color=ORANGE, fontsize=9.5, fontweight="bold")
    big = orders.nlargest(2)
    ax.plot(big.iloc[:1], [0], marker="v", ms=8, color=ORANGE, clip_on=False, zorder=5,
            transform=ax.get_xaxis_transform() if False else ax.transData)
    ax.annotate(f"Largest order £{big.iloc[0]:,.0f}\n(80,995 units, cancelled 12 min later)",
                xy=(big.iloc[0], ytop * 0.02), xytext=(big.iloc[0] / 9, ytop * 0.42), fontsize=8.5, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(0.99, 0.62, f"{n_over_10k} orders above £10k\nkept, not removed", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK2)
    finish(fig, ax, "Most orders are a few hundred pounds; a few very large orders lift the mean",
           f"Distribution of {len(orders):,} orders, Dec 2010 – 9 Dec 2011 (valid merchandise sales, gross of cancellations)",
           "Order values span £0.38 to £168,470, so the axis is logarithmic; no order is excluded. "
           f"The mean (£{mean:,.0f}) sits {mean/med:.1f}× above the median (£{med:,.0f}) because a small number of very large orders pull it up.",
           a.out / "chart1_order_value_histogram")
    pd.DataFrame({"bin_left": bins[:-1], "bin_right": bins[1:],
                  "orders": np.histogram(orders, bins=bins)[0]}).to_csv(a.out / "chart_data_1_order_value_bins.csv", index=False)
    caps.append(("Chart 1 — Order-value histogram", "chart1_order_value_histogram",
                 dict(orders=len(orders), median=med, mean=mean, over_10k=n_over_10k,
                      p25=orders.quantile(.25), p75=orders.quantile(.75), share_100_1000=((orders >= 100) & (orders <= 1000)).mean())))

    # ================= Chart 2: top non-UK markets (sorted horizontal bar) =================
    nu = v2[~v2["Country"].eq(UK)]
    ctry = pd.DataFrame({"net": nu.groupby("Country")["line_revenue"].sum(),
                         "customers": v1[~v1["Country"].eq(UK)].groupby("Country")["CustomerID"].nunique()})
    cn = nu[nu["CustomerID"].notna()].groupby(["Country", "CustomerID"])["line_revenue"].sum()
    ctry["top_share"] = cn.groupby("Country").max() / ctry["net"] * 100
    ctry["orders"] = v1[~v1["Country"].eq(UK)].groupby("Country")["InvoiceNo"].nunique()
    nonuk_total = ctry["net"].sum()
    top = ctry.sort_values("net", ascending=False).head(10)
    top["pct_non_uk"] = top["net"] / nonuk_total * 100
    FEW = 10
    top["concentrated"] = top["customers"] <= FEW
    plot = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    colors = [ORANGE if c else BLUE for c in plot["concentrated"]]
    ax.barh(range(len(plot)), plot["net"], color=colors, height=0.68, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels([f"{c}  ({int(n)} cust.)" for c, n in zip(plot.index, plot["customers"])])
    ax.xaxis.set_major_formatter(mt.FuncFormatter(gbp))
    ax.set_xlabel("Net merchandise revenue (£, after product cancellations)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    xmax = plot["net"].max()
    for i, (v, p, ts, conc) in enumerate(zip(plot["net"], plot["pct_non_uk"], plot["top_share"], plot["concentrated"])):
        lab = f"£{v/1e3:,.0f}k · {p:.1f}%" + (f"  · largest customer {ts:.0f}%" if conc else "")
        ax.text(v + xmax * 0.01, i, lab, va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, xmax * 1.42)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label=f"Broad market (more than {FEW} customers)"),
                       Patch(color=ORANGE, label=f"Few accounts ({FEW} or fewer customers)")],
              loc="lower right", frameon=False, fontsize=8.5)
    finish(fig, ax, "Five markets earn 71% of non-UK revenue; three rely on ≤10 customers",
           f"Top 10 of 37 non-UK markets by net merchandise revenue, Dec 2010 – 9 Dec 2011 (non-UK total £{nonuk_total/1e6:.2f}M)",
           "Labels show net revenue and share of non-UK revenue; orange bars mark markets with 10 or fewer customers, where one account can dominate (98% of revenue in the Netherlands). "
           "Germany and France are the only top-five markets with a broad base of about 90 customers each.",
           a.out / "chart2_top_non_uk_markets")
    top.reset_index().to_csv(a.out / "chart_data_2_top_non_uk.csv", index=False)
    caps.append(("Chart 2 — Top non-UK markets", "chart2_top_non_uk_markets",
                 dict(top=top, nonuk_total=nonuk_total, top5=top.head(5)["pct_non_uk"].sum())))

    # ================= Chart 3: monthly net revenue line =================
    m = v2.assign(month=v2["InvoiceDate"].dt.to_period("M")).groupby("month").agg(
        net=("line_revenue", "sum"), days=("InvoiceDate", lambda s: s.dt.date.nunique()))
    m["per_day"] = m["net"] / m["days"]
    x = np.arange(len(m))
    full, last = slice(0, len(m) - 1), len(m) - 1
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(x[full], m["net"].iloc[full], color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.plot(x[-2:], m["net"].iloc[-2:], color=BLUE, lw=2, ls=(0, (3, 2)))
    ax.plot(x[last], m["net"].iloc[last], marker="o", ms=8, mfc=SURFACE, mec=BLUE, mew=2)
    ax.set_xticks(x)
    ax.set_xticklabels([p.strftime("%b\n%Y") if p.month in (12, 1) else p.strftime("%b") for p in m.index])
    ax.yaxis.set_major_formatter(mt.FuncFormatter(gbp))
    ax.set_ylim(0, m["net"].max() * 1.18)
    ax.set_ylabel("Net merchandise revenue per month (£)")
    ax.set_xlabel("Month of invoice")
    ax.grid(axis="x", visible=False)
    pk = m["net"].idxmax()
    ax.annotate(f"Peak: {pk.strftime('%b %Y')} £{m.loc[pk,'net']/1e6:.2f}M", xy=(x[m.index.get_loc(pk)], m.loc[pk, "net"]),
                xytext=(-10, 12), textcoords="offset points", ha="right", fontsize=9, color=INK, fontweight="bold")
    ax.annotate(f"PARTIAL MONTH\n1–9 Dec 2011, {int(m['days'].iloc[last])} trading days\n"
                f"£{m['per_day'].iloc[last]/1e3:,.0f}k per trading day\n(Nov: £{m['per_day'].iloc[last-1]/1e3:,.0f}k)",
                xy=(x[last], m["net"].iloc[last]), xytext=(-8, -66), textcoords="offset points", ha="right",
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    jan = m.index.get_loc(pd.Period("2011-01", "M"))
    ax.annotate("Jan includes the cancelled\n74,215-jar order (−£77k)", xy=(x[jan], m["net"].iloc[jan]),
                xytext=(18, -46), textcoords="offset points", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    finish(fig, ax, "Net revenue climbs from September into a November peak",
           "Monthly net merchandise revenue (sales less product cancellations), Dec 2010 – 9 Dec 2011",
           "December 2011 covers only 1–9 December and is drawn hollow with a dashed link; its daily rate matches November, so the drop is a data cut-off, not a collapse. "
           "Only one year of data is available, so the autumn rise cannot yet be confirmed as a repeating seasonal pattern.",
           a.out / "chart3_monthly_net_revenue")
    m.assign(partial_month=[False] * (len(m) - 1) + [True]).reset_index().assign(month=lambda d: d["month"].astype(str)) \
        .to_csv(a.out / "chart_data_3_monthly.csv", index=False)
    caps.append(("Chart 3 — Monthly net revenue", "chart3_monthly_net_revenue", dict(m=m)))

    # ================= Chart 4: Quantity vs UnitPrice (log-log density + highlighted extremes) =================
    q, p = v1["Quantity"].astype(float), v1["UnitPrice"].astype(float)
    rho = stats.spearmanr(q, p)[0]
    r_raw = stats.pearsonr(q, p)[0]
    r_log = stats.pearsonr(np.log10(q), np.log10(p))[0]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    hb = ax.hexbin(np.log10(q), np.log10(p), gridsize=70, cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0.2, edgecolors=SURFACE)
    ext = v1["qty_extreme"].to_numpy()
    ax.scatter(np.log10(q[ext]), np.log10(p[ext]), s=14, facecolors="none", edgecolors=ORANGE, linewidths=0.9,
               label=f"Extreme-quantity line (kept), n = {int(ext.sum()):,}")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Order lines per hexagon (log scale)", color=INK2)
    cb.outline.set_visible(False)
    def pow10(v, _):
        val = 10 ** v
        return f"{val:,.0f}" if val >= 1 else f"{val:.2g}"
    ax.xaxis.set_major_locator(mt.FixedLocator(np.log10([1, 10, 100, 1000, 10000, 100000])))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(pow10))
    ax.yaxis.set_major_locator(mt.FixedLocator(np.log10([0.001, 0.01, 0.1, 1, 10, 100, 1000])))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: (f"£{10**v:.3f}" if 10 ** v < 0.01 else f"£{10**v:,.2f}") if 10 ** v < 1 else f"£{10**v:,.0f}"))
    ax.set_xlabel("Quantity per order line (units, log scale)")
    ax.set_ylabel("Unit price (£, log scale)")
    for i, off in zip(v1["Quantity"].nlargest(2).index, [(-70, 34), (-70, -40)]):
        ax.annotate(f"{int(v1.at[i,'Quantity']):,} × £{v1.at[i,'UnitPrice']:.2f}\n(sale later cancelled)",
                    xy=(np.log10(v1.at[i, "Quantity"]), np.log10(v1.at[i, "UnitPrice"])),
                    xytext=off, textcoords="offset points", fontsize=8, color=INK2, ha="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(np.log10(0.0006), np.log10(1500))
    tiny = v1[v1["UnitPrice"] < 0.01]
    if len(tiny):
        ax.scatter(np.log10(tiny["Quantity"]), np.log10(tiny["UnitPrice"]), s=16, color=BLUE, zorder=4)
        ax.annotate(f"{len(tiny)} lines at £{tiny['UnitPrice'].min():.3f}\n(cushion pads, kept)",
                    xy=(np.log10(tiny["Quantity"].median()), np.log10(tiny["UnitPrice"].min())),
                    xytext=(28, 4), textcoords="offset points", fontsize=8, color=INK2, va="center")
    mn = lambda v, d: f"{v:.{d}f}".replace("-", "−")
    ax.text(0.16, 0.975, f"Spearman ρ = {mn(rho, 3)}\nPearson r (log–log) = {mn(r_log, 2)}\nPearson r (raw units) = {mn(r_raw, 3)}",
            transform=ax.transAxes, fontsize=9, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    finish(fig, ax, "Larger purchase quantities are associated with lower unit prices",
           f"Quantity vs unit price for {len(v1):,} valid sales lines, Dec 2010 – 9 Dec 2011 (density on log–log axes)",
           "Both axes are logarithmic and density is shown as hexagons because 527,723 points overplot; all lines are kept, with the 1,501 extreme-quantity lines outlined. "
           f"The raw Pearson correlation is close to zero (r = {r_raw:.3f}), partly reflecting the influence of extreme positive sales quantities; "
           f"Spearman's correlation (ρ = {rho:.3f}) indicates a moderate negative rank association.".replace("-0.", "−0."),
           a.out / "chart4_quantity_vs_unit_price")
    caps.append(("Chart 4 — Quantity vs UnitPrice", "chart4_quantity_vs_unit_price", dict(rho=rho, r_raw=r_raw, r_log=r_log)))

    assert pd.util.hash_pandas_object(df, index=True).sum() == h0, "data modified"
    print("charts written to", a.out.resolve())
    return caps


if __name__ == "__main__":
    main()
