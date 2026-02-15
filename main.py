"""
Expense Tracker — Auto-categorize & Visualize Your Spending
============================================================
Portfolio Piece #2 by Ashvik Sinha

Interactive terminal app powered by Rich.  Reads a CSV of expenses,
auto-categorizes uncategorized rows via keyword matching, displays
beautiful tables and summaries, generates bar + pie charts, and
exports everything to an output/ folder.

Usage:
    python main.py                      # interactive menu
    python main.py path/to/expenses.csv # pre-load a specific file
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TaskProgressColumn, TimeElapsedColumn,
)
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich import box


# ──────────────────────────────────────────────────────────────────────
# RICH CONSOLE  (single instance used everywhere)
# ──────────────────────────────────────────────────────────────────────

console = Console()


# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

DEFAULT_CSV = "sample_expenses.csv"
OUTPUT_DIR = Path("output")

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Food":          ["restaurant", "cafe", "swiggy", "zomato", "food",
                      "grocery", "meal", "pizza", "burger", "bakery",
                      "dining", "lunch", "dinner", "breakfast"],
    "Transport":     ["uber", "ola", "petrol", "metro", "bus", "auto",
                      "parking", "cab", "fuel", "toll", "train", "flight"],
    "Shopping":      ["amazon", "flipkart", "myntra", "shop", "clothes",
                      "electronics", "mall", "shoes", "fashion"],
    "Bills":         ["electricity", "wifi", "rent", "phone", "recharge",
                      "subscription", "water", "gas", "insurance",
                      "internet", "broadband"],
    "Entertainment": ["movie", "netflix", "spotify", "game", "concert",
                      "disney", "hotstar", "prime", "youtube", "event"],
    "Health":        ["medicine", "doctor", "pharmacy", "hospital", "gym",
                      "clinic", "dental", "therapy", "health", "medical"],
    "Education":     ["course", "book", "udemy", "fees", "college",
                      "tuition", "school", "exam", "coursera", "library"],
}

FALLBACK_CATEGORY = "Other"

# Emoji + Rich colour style for each category
CATEGORY_STYLE: dict[str, tuple[str, str]] = {
    "Food":          ("🍔", "green"),
    "Transport":     ("🚗", "blue"),
    "Shopping":      ("🛍️ ", "magenta"),
    "Bills":         ("📄", "yellow"),
    "Entertainment": ("🎬", "cyan"),
    "Health":        ("💊", "red"),
    "Education":     ("📚", "bright_blue"),
    "Other":         ("📦", "white"),
}

# Matplotlib chart colour palette — professional, muted tones
CHART_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

# Map each category to a fixed colour for consistency across all charts
CATEGORY_COLOURS: dict[str, str] = {
    "Food":          "#FF6B6B",
    "Transport":     "#4ECDC4",
    "Shopping":      "#45B7D1",
    "Bills":         "#FFA07A",
    "Entertainment": "#F7DC6F",
    "Health":        "#BB8FCE",
    "Education":     "#85C1E9",
    "Other":         "#98D8C8",
}

BRAND_NAME = "Expense Tracker by Ashvik Sinha"

# Currency symbol for charts (₹ may not render in all matplotlib fonts)
_CHART_CURRENCY = "Rs."


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def _cat_label(category: str) -> str:
    """Return 'emoji Name' for a category, e.g. '🍔 Food'."""
    emoji, _ = CATEGORY_STYLE.get(category, ("📦", "white"))
    return f"{emoji} {category}"


def _cat_style(category: str) -> str:
    """Return the Rich colour name for a category."""
    _, style = CATEGORY_STYLE.get(category, ("📦", "white"))
    return style


def _amount_style(amount: float, high: float, low: float) -> str:
    """Colour-code an amount: bold red if top-quartile, green if bottom."""
    threshold_high = low + 0.75 * (high - low)
    threshold_low = low + 0.25 * (high - low)
    if amount >= threshold_high:
        return "bold red"
    if amount <= threshold_low:
        return "green"
    return "white"


def _fmt_currency(amount: float) -> str:
    """Format a number as ₹xx,xxx.xx"""
    return f"₹{amount:,.2f}"


# ──────────────────────────────────────────────────────────────────────
# 1. READ CSV
# ──────────────────────────────────────────────────────────────────────

def read_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Read an expense CSV and return a cleaned DataFrame.

    Expected columns: Date, Description, Amount, Category (optional).
    Raises clear errors for missing files, empty data, or bad columns.
    """
    filepath = Path(filepath)

    # --- file existence ---
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: [bold]{filepath.resolve()}[/bold]"
        )
    if filepath.stat().st_size == 0:
        raise ValueError(
            f"File is empty: [bold]{filepath.resolve()}[/bold]"
        )

    file_size = filepath.stat().st_size
    size_label = (
        f"{file_size:,} bytes"
        if file_size < 1024
        else f"{file_size / 1024:.1f} KB"
    )

    # --- read with a progress bar ---
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]Loading CSV…"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("read", total=4)

        df = pd.read_csv(filepath)
        progress.advance(task)

        if df.empty:
            raise ValueError("CSV loaded but contains no data rows.")

        # --- column validation ---
        required = {"Date", "Description", "Amount"}
        actual = set(df.columns.str.strip())
        missing = required - actual
        if missing:
            raise ValueError(
                f"Missing required columns: "
                f"[bold]{', '.join(sorted(missing))}[/bold].\n"
                f"  Found: {', '.join(df.columns.tolist())}"
            )

        df.columns = df.columns.str.strip()
        if "Category" not in df.columns:
            df["Category"] = ""
        progress.advance(task)

        # --- type cleanup ---
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        invalid_amounts = df["Amount"].isna().sum()
        if invalid_amounts:
            console.print(
                f"  [yellow]⚠ Dropped {invalid_amounts} row(s) with "
                f"non-numeric amounts.[/yellow]"
            )
            df = df.dropna(subset=["Amount"])
        progress.advance(task)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        df["Description"] = df["Description"].astype(str).str.strip()
        df["Category"] = df["Category"].astype(str).str.strip()
        progress.advance(task)

    # --- success banner ---
    console.print(
        Panel(
            f"[bold green]✓[/bold green]  Loaded [bold]{filepath.name}[/bold]\n"
            f"   Rows: [cyan]{len(df)}[/cyan]   •   "
            f"Size: [cyan]{size_label}[/cyan]   •   "
            f"Columns: [cyan]{', '.join(df.columns)}[/cyan]",
            border_style="green",
            padding=(0, 2),
        )
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. AUTO-CATEGORIZE
# ──────────────────────────────────────────────────────────────────────

def categorize_expense(description: str) -> str:
    """
    Return a category for a single expense description using keyword
    matching.  Case-insensitive; returns 'Other' when no match is found.
    """
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return FALLBACK_CATEGORY


def auto_categorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in empty / NaN categories using keyword-based classification.
    Already-categorized rows are left untouched.
    """
    mask = (
        df["Category"].isna()
        | df["Category"].isin(["", "nan", "NaN", "None", "none"])
    )
    to_classify = mask.sum()

    if to_classify == 0:
        console.print(
            "  [dim]ℹ  All expenses already categorized — nothing to do.[/dim]\n"
        )
        return df

    # --- progress bar per-row ---
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold cyan]Categorizing expenses…"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("categorize", total=to_classify)
        results: list[str] = []
        for desc in df.loc[mask, "Description"]:
            results.append(categorize_expense(desc))
            progress.advance(task)
        df.loc[mask, "Category"] = results

    console.print(
        f"  [bold green]✓[/bold green]  Auto-categorized "
        f"[bold]{to_classify}[/bold] expense(s).\n"
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# 3. GENERATE SUMMARY
# ──────────────────────────────────────────────────────────────────────

def generate_summary(df: pd.DataFrame) -> dict:
    """
    Display a beautiful Rich summary and return the numbers as a dict.
    """
    total = df["Amount"].sum()
    avg = df["Amount"].mean()
    by_category = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )
    top_5 = df.nlargest(5, "Amount")[
        ["Date", "Description", "Amount", "Category"]
    ]

    # --- Header panel ---
    period = ""
    if not df["Date"].isna().all():
        d_min = df["Date"].min().strftime("%d %b %Y")
        d_max = df["Date"].max().strftime("%d %b %Y")
        period = f"   Period : [cyan]{d_min}[/cyan] → [cyan]{d_max}[/cyan]\n"

    console.print()
    console.print(
        Panel(
            f"[bold]Total Spent[/bold] : [bold yellow]{_fmt_currency(total)}[/bold yellow]\n"
            f"   Transactions : [cyan]{len(df)}[/cyan]\n"
            f"   Average      : [cyan]{_fmt_currency(avg)}[/cyan]\n"
            f"{period}",
            title="[bold]📊  Expense Summary[/bold]",
            border_style="bright_cyan",
            padding=(1, 3),
        )
    )

    # --- Category breakdown table ---
    cat_table = Table(
        title="Spending by Category",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold",
        header_style="bold bright_white on dark_green",
        padding=(0, 1),
    )
    cat_table.add_column("Category", min_width=18)
    cat_table.add_column("Amount", justify="right", min_width=12)
    cat_table.add_column("Share", justify="right", min_width=8)
    cat_table.add_column("", min_width=20)  # inline bar

    max_amt = by_category.max() if not by_category.empty else 1
    for cat, amt in by_category.items():
        pct = amt / total * 100
        bar_len = int(amt / max_amt * 16)
        bar_char = "█" * bar_len + "░" * (16 - bar_len)
        style = _cat_style(cat)
        cat_table.add_row(
            f"[{style}]{_cat_label(cat)}[/{style}]",
            f"[bold]{_fmt_currency(amt)}[/bold]",
            f"{pct:.1f}%",
            f"[{style}]{bar_char}[/{style}]",
        )

    console.print(cat_table)
    console.print()

    # --- Top 5 expenses table ---
    top_table = Table(
        title="🔝 Top 5 Expenses",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        title_style="bold",
        header_style="bold bright_white on grey37",
        padding=(0, 1),
    )
    top_table.add_column("#", justify="center", width=3)
    top_table.add_column("Date", min_width=12)
    top_table.add_column("Description", min_width=26)
    top_table.add_column("Amount", justify="right", min_width=12)
    top_table.add_column("Category", min_width=16)

    amt_high = df["Amount"].max()
    amt_low = df["Amount"].min()
    for rank, (_, row) in enumerate(top_5.iterrows(), start=1):
        date_str = (
            row["Date"].strftime("%d %b %Y") if pd.notna(row["Date"]) else "—"
        )
        style = _amount_style(row["Amount"], amt_high, amt_low)
        cat_style = _cat_style(row["Category"])
        top_table.add_row(
            str(rank),
            date_str,
            row["Description"],
            f"[{style}]{_fmt_currency(row['Amount'])}[/{style}]",
            f"[{cat_style}]{_cat_label(row['Category'])}[/{cat_style}]",
        )

    console.print(top_table)
    console.print()

    return {
        "total": total,
        "by_category": by_category,
        "top_5": top_5,
    }


# ──────────────────────────────────────────────────────────────────────
# 4. CREATE CHARTS  —  Professional Dashboard System
# ──────────────────────────────────────────────────────────────────────

def _get_cat_colour(category: str) -> str:
    """Return a consistent hex colour for a category."""
    return CATEGORY_COLOURS.get(
        category,
        CHART_PALETTE[hash(category) % len(CHART_PALETTE)],
    )


def _apply_chart_style() -> None:
    """Apply a clean, modern style globally for all charts."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("ggplot")
        except OSError:
            pass  # fall back to default

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "savefig.facecolor": "#FAFAFA",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def _add_branding(fig: plt.Figure, timestamp: str) -> None:
    """Add brand watermark and generation timestamp to a figure."""
    fig.text(
        0.99, 0.005,
        f"{BRAND_NAME}  •  Generated {timestamp}",
        ha="right", va="bottom",
        fontsize=7, color="#AAAAAA", style="italic",
    )


# ── 4a. Bar Chart ────────────────────────────────────────────────────

def _chart_bar(
    ax: plt.Axes,
    by_category: pd.Series,
) -> None:
    """Horizontal bar chart — spending by category, highest to lowest."""
    cats = by_category.index.tolist()[::-1]       # lowest at top → highest at bottom
    vals = by_category.values[::-1]
    colours = [_get_cat_colour(c) for c in cats]

    bars = ax.barh(cats, vals, color=colours, height=0.6, edgecolor="white", linewidth=0.8)

    # amount labels on bars
    ax.bar_label(
        bars,
        labels=[f" {_CHART_CURRENCY}{v:,.0f}" for v in vals],
        padding=6, fontsize=9, fontweight="bold",
    )

    # average spending line
    avg = float(by_category.mean())
    ax.axvline(avg, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        avg, len(cats) - 0.3,
        f"  Avg {_CHART_CURRENCY}{avg:,.0f}",
        color="#E74C3C", fontsize=8, fontweight="bold", va="bottom",
    )

    ax.set_xlabel(f"Amount ({_CHART_CURRENCY})", fontsize=11)
    ax.set_title("Spending by Category", pad=12)
    ax.set_xlim(right=float(by_category.max()) * 1.25)
    ax.tick_params(axis="y", labelsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{_CHART_CURRENCY}{x:,.0f}"))


# ── 4b. Pie Chart ────────────────────────────────────────────────────

def _chart_pie(
    ax: plt.Axes,
    by_category: pd.Series,
) -> None:
    """Pie chart with exploded largest slice, shadow, and amount labels."""
    cats = by_category.index.tolist()
    vals = by_category.values
    colours = [_get_cat_colour(c) for c in cats]
    total = float(vals.sum())

    # explode the largest slice
    explode = [0.06 if v == vals.max() else 0.0 for v in vals]

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=cats,
        autopct=lambda pct: f"{pct:.1f}%\n{_CHART_CURRENCY}{total * pct / 100:,.0f}",
        startangle=140,
        colors=colours,
        explode=explode,
        pctdistance=0.75,
        shadow=True,
        wedgeprops={"edgecolor": "white", "linewidth": 1.8},
    )
    for t in texts:
        t.set_fontsize(9)
    for t in autotexts:
        t.set_fontsize(7.5)
        t.set_fontweight("bold")
        t.set_color("#333333")

    ax.set_title("Expense Breakdown", pad=16)


# ── 4c. Time-Trend Line Chart ────────────────────────────────────────

def _chart_trend(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """
    Daily spending line chart with markers, moving average, and
    highlighted high-spending days.
    """
    daily = (
        df.dropna(subset=["Date"])
        .groupby(df["Date"].dt.date)["Amount"]
        .sum()
        .sort_index()
    )
    if daily.empty:
        ax.text(0.5, 0.5, "No date data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999999")
        ax.set_title("Spending Over Time", pad=12)
        return

    dates = pd.to_datetime(daily.index)
    amounts = daily.values.astype(float)

    # main line
    ax.plot(
        dates, amounts,
        color="#45B7D1", linewidth=2, marker="o", markersize=5,
        markerfacecolor="white", markeredgecolor="#45B7D1", markeredgewidth=1.5,
        label="Daily Spend", zorder=3,
    )

    # 3-day moving average (if enough data)
    window = min(3, len(amounts))
    if window >= 2:
        ma = pd.Series(amounts).rolling(window, min_periods=1).mean().values
        ax.plot(
            dates, ma,
            color="#E74C3C", linewidth=1.5, linestyle="--",
            alpha=0.7, label=f"{window}-day Avg", zorder=2,
        )

    # highlight high-spending days (above 75th percentile)
    p75 = float(np.percentile(amounts, 75))
    high_mask = amounts >= p75
    if high_mask.any():
        ax.scatter(
            dates[high_mask], amounts[high_mask],
            color="#FF6B6B", s=90, zorder=4,
            edgecolors="white", linewidths=1.2,
            label=f"High (>= {_CHART_CURRENCY}{p75:,.0f})",
        )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Amount ({_CHART_CURRENCY})", fontsize=11)
    ax.set_title("Spending Over Time", pad=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{_CHART_CURRENCY}{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)


# ── 4d. Calendar Heatmap ─────────────────────────────────────────────

def _chart_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """
    Calendar-style heatmap of daily spending intensity.
    Falls back to a weekly-category heatmap if date span < 30 days.
    """
    date_df = df.dropna(subset=["Date"]).copy()
    if date_df.empty:
        ax.text(0.5, 0.5, "No date data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999999")
        ax.set_title("Spending Heatmap", pad=12)
        return

    date_span = (date_df["Date"].max() - date_df["Date"].min()).days

    if date_span >= 30:
        # ── Calendar heatmap: weeks × weekdays ──
        daily = date_df.groupby(date_df["Date"].dt.date)["Amount"].sum()
        full_range = pd.date_range(daily.index.min(), daily.index.max())
        daily = daily.reindex(full_range, fill_value=0)

        week_labels = [d.strftime("%d %b") for d in daily.index[::7]]
        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        n_weeks = int(np.ceil(len(daily) / 7))
        padded = np.full(n_weeks * 7, np.nan)
        padded[: len(daily)] = daily.values
        matrix = padded.reshape(n_weeks, 7).T  # 7 rows (weekdays) × n_weeks cols

        sns.heatmap(
            matrix, ax=ax,
            cmap="YlOrRd", linewidths=1, linecolor="white",
            cbar_kws={"label": f"{_CHART_CURRENCY} Spent", "shrink": 0.6},
            annot=False, mask=np.isnan(matrix),
            xticklabels=week_labels[:n_weeks] if n_weeks <= 15 else False,
            yticklabels=weekday_labels,
        )
        ax.set_title("Daily Spending Intensity", pad=12)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
    else:
        # ── Category × Day heatmap for shorter spans ──
        date_df["DayLabel"] = date_df["Date"].dt.strftime("%d %b")
        # keep original dates for sorting
        day_order = (
            date_df[["Date", "DayLabel"]]
            .drop_duplicates("DayLabel")
            .sort_values("Date")["DayLabel"]
            .tolist()
        )
        pivot = date_df.pivot_table(
            index="Category", columns="DayLabel",
            values="Amount", aggfunc="sum", fill_value=0,
        )
        # sort columns chronologically using original dates
        pivot = pivot.reindex(columns=[c for c in day_order if c in pivot.columns])

        sns.heatmap(
            pivot, ax=ax,
            cmap="YlOrRd", linewidths=0.8, linecolor="white",
            cbar_kws={"label": f"{_CHART_CURRENCY} Spent", "shrink": 0.6},
            annot=True, fmt=".0f", annot_kws={"fontsize": 6},
        )
        ax.set_title("Category × Day Spending", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)


# ── Master chart orchestrator ────────────────────────────────────────

def create_charts(
    by_category: pd.Series,
    output_dir: Path,
    df: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path]:
    """
    Generate and save professional charts:
      • Individual: category_chart.png, pie_chart.png
      • Dashboard:  dashboard.png  (2×2 grid with all four charts)

    Returns (bar_path, pie_path, dashboard_path).
    """
    if by_category.empty:
        console.print("  [yellow]⚠  No category data — skipping charts.[/yellow]")
        return Path(), Path(), Path()

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_chart_style()
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    # --- spinner while rendering ---
    with console.status(
        "[bold cyan]Rendering professional charts…[/bold cyan]", spinner="aesthetic"
    ):
        # ═══════════════════════════════════════════════════════════
        # Individual chart 1 — Bar
        # ═══════════════════════════════════════════════════════════
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5.5))
        _chart_bar(ax_bar, by_category)
        _add_branding(fig_bar, timestamp)
        fig_bar.tight_layout(rect=[0, 0.02, 1, 1])
        bar_path = output_dir / "category_chart.png"
        fig_bar.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.close(fig_bar)

        # ═══════════════════════════════════════════════════════════
        # Individual chart 2 — Pie
        # ═══════════════════════════════════════════════════════════
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        _chart_pie(ax_pie, by_category)
        _add_branding(fig_pie, timestamp)
        fig_pie.tight_layout(rect=[0, 0.02, 1, 1])
        pie_path = output_dir / "pie_chart.png"
        fig_pie.savefig(pie_path, dpi=200, bbox_inches="tight")
        plt.close(fig_pie)

        # ═══════════════════════════════════════════════════════════
        # Dashboard — 2×2 grid
        # ═══════════════════════════════════════════════════════════
        fig_dash, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig_dash.patch.set_facecolor("#FAFAFA")

        # Top-left: Bar chart
        _chart_bar(axes[0, 0], by_category)

        # Top-right: Pie chart
        _chart_pie(axes[0, 1], by_category)

        # Bottom-left: Time trend (needs full df)
        if df is not None and "Date" in df.columns:
            _chart_trend(axes[1, 0], df)
        else:
            axes[1, 0].text(
                0.5, 0.5, "Time trend requires date data",
                transform=axes[1, 0].transAxes,
                ha="center", va="center", fontsize=12, color="#999999",
            )
            axes[1, 0].set_title("Spending Over Time", pad=12)

        # Bottom-right: Heatmap (needs full df)
        if df is not None and "Date" in df.columns:
            _chart_heatmap(axes[1, 1], df)
        else:
            axes[1, 1].text(
                0.5, 0.5, "Heatmap requires date data",
                transform=axes[1, 1].transAxes,
                ha="center", va="center", fontsize=12, color="#999999",
            )
            axes[1, 1].set_title("Spending Heatmap", pad=12)

        # ── Dashboard header ──
        total = float(by_category.sum())
        if df is not None and not df["Date"].isna().all():
            d_min = df["Date"].min().strftime("%d %b %Y")
            d_max = df["Date"].max().strftime("%d %b %Y")
            subtitle = f"{d_min}  to  {d_max}   |   Total: {_CHART_CURRENCY}{total:,.0f}"
        else:
            subtitle = f"Total: {_CHART_CURRENCY}{total:,.0f}"

        fig_dash.suptitle(
            "Expense Dashboard",
            fontsize=22, fontweight="bold", color="#2C3E50",
            y=0.98,
        )
        fig_dash.text(
            0.5, 0.955, subtitle,
            ha="center", fontsize=12, color="#7F8C8D",
        )

        _add_branding(fig_dash, timestamp)
        fig_dash.tight_layout(rect=[0, 0.02, 1, 0.94])

        dash_path = output_dir / "dashboard.png"
        fig_dash.savefig(dash_path, dpi=300, bbox_inches="tight")
        plt.close(fig_dash)

    console.print(
        f"  [bold green]✓[/bold green]  Bar chart   → [link=file://{bar_path.resolve()}]{bar_path}[/link]\n"
        f"  [bold green]✓[/bold green]  Pie chart   → [link=file://{pie_path.resolve()}]{pie_path}[/link]\n"
        f"  [bold green]✓[/bold green]  Dashboard   → [link=file://{dash_path.resolve()}]{dash_path}[/link]  [dim](300 dpi)[/dim]"
    )
    return bar_path, pie_path, dash_path


# ──────────────────────────────────────────────────────────────────────
# 5. SAVE OUTPUT
# ──────────────────────────────────────────────────────────────────────

def save_output(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Export the categorized DataFrame to a clean CSV inside output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "categorized_expenses.csv"

    export_df = df.copy()
    export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
    export_df.to_csv(out_path, index=False)

    console.print(
        f"  [bold green]✓[/bold green]  CSV saved  → "
        f"[link=file://{out_path.resolve()}]{out_path}[/link]\n"
    )
    return out_path


# ──────────────────────────────────────────────────────────────────────
# 6. ADD CUSTOM CATEGORY RULES  (interactive)
# ──────────────────────────────────────────────────────────────────────

def add_custom_rules() -> None:
    """
    Let the user add keyword → category mappings at runtime.
    New rules persist for the current session only.
    """
    console.print(
        Panel(
            "[dim]Add keywords to an existing category, or create a new one.\n"
            "Type [bold]done[/bold] when finished.[/dim]",
            title="[bold]🛠  Custom Category Rules[/bold]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )

    while True:
        keyword = Prompt.ask(
            "  [cyan]Keyword[/cyan] (or [bold]done[/bold])"
        ).strip().lower()
        if keyword in ("done", ""):
            break

        # show existing categories as numbered choices
        cats = list(CATEGORY_KEYWORDS.keys()) + [FALLBACK_CATEGORY]
        for i, c in enumerate(cats, 1):
            console.print(f"    [dim]{i}.[/dim] {_cat_label(c)}")
        console.print(f"    [dim]{len(cats) + 1}.[/dim] ✨ New category")

        choice = IntPrompt.ask(
            "  [cyan]Category #[/cyan]",
            default=1,
        )

        if choice == len(cats) + 1:
            new_cat = Prompt.ask("  [cyan]New category name[/cyan]").strip()
            if new_cat:
                CATEGORY_KEYWORDS[new_cat] = [keyword]
                # register a default style so it renders nicely
                CATEGORY_STYLE.setdefault(new_cat, ("🏷️ ", "bright_yellow"))
                console.print(
                    f"  [green]✓[/green]  Created [bold]{new_cat}[/bold] "
                    f"with keyword [bold]'{keyword}'[/bold].\n"
                )
        elif 1 <= choice <= len(cats):
            target = cats[choice - 1]
            if target == FALLBACK_CATEGORY:
                console.print("  [yellow]⚠  Cannot add keywords to 'Other'.[/yellow]\n")
                continue
            CATEGORY_KEYWORDS[target].append(keyword)
            console.print(
                f"  [green]✓[/green]  Added [bold]'{keyword}'[/bold] → "
                f"{_cat_label(target)}.\n"
            )
        else:
            console.print("  [red]✗  Invalid choice.[/red]\n")


# ──────────────────────────────────────────────────────────────────────
# WELCOME BANNER
# ──────────────────────────────────────────────────────────────────────

BANNER = r"""
[bold bright_cyan]
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║     ███████╗██╗  ██╗██████╗ ███████╗███╗   ██╗███████╗   ║
  ║     ██╔════╝╚██╗██╔╝██╔══██╗██╔════╝████╗  ██║██╔════╝   ║
  ║     █████╗   ╚███╔╝ ██████╔╝█████╗  ██╔██╗ ██║███████╗   ║
  ║     ██╔══╝   ██╔██╗ ██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║   ║
  ║     ███████╗██╔╝ ██╗██║     ███████╗██║ ╚████║███████║   ║
  ║     ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝   ║
  ║                                                           ║
  ║        [bold bright_yellow]  💰  T R A C K E R   v2.0[/bold bright_yellow]                       ║
  ║        [dim]Auto-categorize & visualize spending[/dim]              ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
[/bold bright_cyan]"""


# ──────────────────────────────────────────────────────────────────────
# INTERACTIVE MENU
# ──────────────────────────────────────────────────────────────────────

def _build_menu() -> Table:
    """Build the main-menu table (no outer borders, clean look)."""
    menu = Table(
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
        highlight=True,
    )
    menu.add_column(justify="center", width=4)
    menu.add_column(min_width=36)

    items = [
        ("1", "[bold green]Load & categorize expenses[/bold green]"),
        ("2", "[bold cyan]View summary statistics[/bold cyan]"),
        ("3", "[bold yellow]Generate charts[/bold yellow]"),
        ("4", "[bold magenta]Export categorized CSV[/bold magenta]"),
        ("5", "[bold bright_blue]Add custom category rules[/bold bright_blue]"),
        ("6", "[bold red]Exit[/bold red]"),
    ]
    for num, label in items:
        menu.add_row(f"[bold bright_white][{num}][/bold bright_white]", label)
    return menu


def main() -> None:
    """Interactive menu loop — the program's entry-point."""

    console.print(BANNER)

    # ── state ──
    df: pd.DataFrame | None = None
    summary: dict | None = None

    # pre-load if a file was passed via CLI
    csv_path: str | None = sys.argv[1] if len(sys.argv) > 1 else None

    while True:
        console.print(
            Panel(
                _build_menu(),
                title="[bold]  Main Menu  [/bold]",
                border_style="bright_cyan",
                padding=(1, 2),
            )
        )
        choice = Prompt.ask(
            "[bold bright_white]Select option[/bold bright_white]",
            choices=["1", "2", "3", "4", "5", "6"],
            default="1",
        )
        console.print()

        # ── 1. Load & categorize ─────────────────────────────────────
        if choice == "1":
            if csv_path is None:
                csv_path = Prompt.ask(
                    f"  [cyan]CSV file path[/cyan]",
                    default=DEFAULT_CSV,
                ).strip()
            try:
                df = read_csv(csv_path)
                df = auto_categorize(df)
                # also compute summary right away for convenience
                summary = generate_summary(df)
            except (FileNotFoundError, ValueError) as exc:
                console.print(f"  [bold red]✗[/bold red]  {exc}\n")
                csv_path = None
                df = None
                continue
            # reset so next Load asks again
            csv_path = None

        # ── 2. View summary ──────────────────────────────────────────
        elif choice == "2":
            if df is None:
                console.print(
                    "  [yellow]⚠  No data loaded yet — choose option 1 first.[/yellow]\n"
                )
                continue
            summary = generate_summary(df)

        # ── 3. Generate charts ───────────────────────────────────────
        elif choice == "3":
            if summary is None or df is None:
                console.print(
                    "  [yellow]⚠  Load data first (option 1).[/yellow]\n"
                )
                continue
            create_charts(summary["by_category"], OUTPUT_DIR, df=df)
            console.print()

        # ── 4. Export CSV ────────────────────────────────────────────
        elif choice == "4":
            if df is None:
                console.print(
                    "  [yellow]⚠  Load data first (option 1).[/yellow]\n"
                )
                continue
            save_output(df, OUTPUT_DIR)

        # ── 5. Custom rules ──────────────────────────────────────────
        elif choice == "5":
            add_custom_rules()
            # re-categorize with new rules if data is loaded
            if df is not None:
                console.print(
                    "  [dim]Re-applying categories with updated rules…[/dim]"
                )
                # reset categories that were auto-assigned so new rules apply
                df["Category"] = df["Category"].where(
                    ~df["Category"].isin(
                        list(CATEGORY_KEYWORDS.keys()) + [FALLBACK_CATEGORY]
                    ),
                    other="",
                )
                df = auto_categorize(df)
                summary = generate_summary(df)

        # ── 6. Exit ──────────────────────────────────────────────────
        elif choice == "6":
            console.print(
                Panel(
                    "[bold bright_cyan]Thanks for using Expense Tracker![/bold bright_cyan]\n"
                    "[dim]Output files are in the [bold]output/[/bold] folder.[/dim]",
                    border_style="bright_cyan",
                    padding=(1, 3),
                )
            )
            break


if __name__ == "__main__":
    main()