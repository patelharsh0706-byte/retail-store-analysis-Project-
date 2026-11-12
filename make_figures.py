#!/usr/bin/env python3
"""Compact report figures (Figures 1–5) for the IE5005 PDF. Same data and definitions as
retail_charts.py / retail_rfm.py; sized for half-page width so text stays legible in print.
Titles, axis labels (with units) and legends are inside each figure; numbered captions are in
the report. No record is modified or removed."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent; OUT = ROOT / "figures"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF})
W, H = 3.6, 2.7

def gbp(x, _=None):
    return f"£{x/1e6:.1f}M" if abs(x) >= 1e6 else (f"£{x/1e3:,.0f}k" if abs(x) >= 1e3 else (f"£{x:,.0f}" if x >= 1 else f"£{x:.2g}"))

def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / f"{name}.png", dpi=400); plt.close(fig)

df = pd.read_csv(ROOT / "prepared/retail_prepared.csv.gz", dtype={"InvoiceNo": str, "StockCode": str},
                 usecols=["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "line_revenue", "line_type",
                          "in_merchandise_sales", "qty_extreme"], low_memory=False)
v1 = df[df["in_merchandise_sales"] & df["line_type"].eq("sale")]

# Figure 1 — distribution: order value histogram (log x)
orders = v1.groupby("InvoiceNo")["line_revenue"].sum()
med, mean = orders.median(), orders.mean()
fig, ax = plt.subplots(figsize=(W, H))
ax.hist(orders, bins=np.logspace(np.log10(orders.min()), np.log10(orders.max()), 50), color=BLUE, edgecolor=SURF, linewidth=0.4)
ax.set_xscale("log"); ax.xaxis.set_major_formatter(mt.FuncFormatter(gbp)); ax.grid(axis="x", visible=False)
ax.axvline(med, color=INK, lw=1.2, label=f"Median £{med:,.0f}")
ax.axvline(mean, color=ORANGE, lw=1.2, ls="--", label=f"Mean £{mean:,.0f}")
ax.set_title("Order value: long right tail")
ax.set_xlabel("Order value (£, log scale)"); ax.set_ylabel("Number of orders")
ax.yaxis.set_major_formatter(mt.StrMethodFormatter("{x:,.0f}"))
ax.legend(frameon=False, loc="upper right")
save(fig, "fig1_order_value")

# Figure 2 — ranking: top 10 non-UK markets
top = pd.read_csv(ROOT / "charts/chart_data_2_top_non_uk.csv").head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(W, H))
cols = [ORANGE if c else BLUE for c in top["concentrated"]]
ax.barh(range(len(top)), top["net"], color=cols, height=0.7)
ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{c} ({int(n)})" for c, n in zip(top["Country"], top["customers"])])
for i, v in enumerate(top["net"]):
    ax.text(v + 4000, i, f"£{v/1e3:,.0f}k", va="center", fontsize=6.5, color=INK)
ax.set_xlim(0, top["net"].max() * 1.22); ax.xaxis.set_major_locator(mt.MultipleLocator(100000)); ax.xaxis.set_major_formatter(mt.FuncFormatter(gbp))
ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
ax.set_title("Top 10 non-UK markets")
ax.set_xlabel("Net merchandise revenue (£)"); ax.set_ylabel("Country (customers)")
ax.legend(handles=[Patch(color=BLUE, label="> 10 customers"), Patch(color=ORANGE, label="≤ 10 customers")],
          frameon=False, loc="lower right")
save(fig, "fig2_non_uk")

# Figure 3 — change over time: monthly net revenue
m = pd.read_csv(ROOT / "charts/chart_data_3_monthly.csv")
x = np.arange(len(m)); last = len(m) - 1
fig, ax = plt.subplots(figsize=(W, H))
ax.plot(x[:-1], m["net"].iloc[:-1] / 1e6, color=BLUE, lw=1.6, marker="o", ms=3.5, label="Full month")
ax.plot(x[-2:], m["net"].iloc[-2:] / 1e6, color=BLUE, lw=1.4, ls=(0, (3, 2)))
ax.plot(x[last], m["net"].iloc[last] / 1e6, marker="o", ms=5, mfc=SURF, mec=BLUE, mew=1.5, ls="none",
        label="Partial month (1–9 Dec, 8 trading days)")
labs = pd.PeriodIndex(m["month"], freq="M")
ax.set_xticks(x); ax.set_xticklabels([p.strftime("%b\n%y") if p.month in (12, 1) else p.strftime("%b") for p in labs])
ax.set_ylim(0, 1.75); ax.grid(axis="x", visible=False)
ax.set_title("Monthly net revenue peaks in November 2011")
ax.set_xlabel("Month"); ax.set_ylabel("Net merchandise revenue (£ million)")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig3_monthly")

# Figure 4 — relationship: quantity vs unit price (log–log density)
q, p = v1["Quantity"].astype(float), v1["UnitPrice"].astype(float)
fig, ax = plt.subplots(figsize=(W, H))
cmap = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
hb = ax.hexbin(np.log10(q), np.log10(p), gridsize=45, cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0.1, edgecolors=SURF)
ext = v1["qty_extreme"].to_numpy()
ax.scatter(np.log10(q[ext]), np.log10(p[ext]), s=4, facecolors="none", edgecolors=ORANGE, linewidths=0.5,
           label="Extreme-quantity line (kept)")
cb = fig.colorbar(hb, ax=ax, pad=0.01); cb.set_label("Lines per hexagon (log)", fontsize=6.5); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
ax.xaxis.set_major_locator(mt.FixedLocator(np.log10([1, 10, 100, 1000, 10000, 100000])))
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{10**v:,.0f}"))
ax.yaxis.set_major_locator(mt.FixedLocator(np.log10([0.001, 0.01, 0.1, 1, 10, 100, 1000])))
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: (f"£{10**v:.3g}" if 10 ** v < 1 else f"£{10**v:,.0f}")))
ax.set_ylim(np.log10(0.0006), np.log10(1500))
ax.set_title("Larger quantities, lower unit prices")
ax.set_xlabel("Quantity per line (units, log scale)"); ax.set_ylabel("Unit price (£, log scale)")
ax.legend(frameon=False, loc="upper right", markerscale=2)
save(fig, "fig4_qty_price")

# Figure 5 — RFM segments: share of customers vs share of net revenue
s = pd.read_csv(ROOT / "rfm/segment_summary.csv").head(6).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.6, 2.35))
y = np.arange(len(s)); h = 0.38
ax.barh(y + h / 2, s["pct_customers"], height=h, color=BLUE, label="Share of scored customers (%)")
ax.barh(y - h / 2, s["pct_of_attributable_net"], height=h, color=ORANGE, label="Share of attributable net revenue (%)")
for yi, (c, r) in enumerate(zip(s["pct_customers"], s["pct_of_attributable_net"])):
    ax.text(c + 0.5, yi + h / 2, f"{c:.1f}%", va="center", fontsize=6.3); ax.text(r + 0.5, yi - h / 2, f"{r:.1f}%", va="center", fontsize=6.3)
ax.set_yticks(y); ax.set_yticklabels([f"{a} ({int(n):,})" for a, n in zip(s["segment"], s["customers"])])
ax.set_xlim(0, 75); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
ax.set_title("Champions: 21% of customers, 66% of attributable net revenue")
ax.set_xlabel("Percent (%)"); ax.legend(frameon=False, loc="lower right")
save(fig, "fig5_rfm")
print("figures written to", OUT)
