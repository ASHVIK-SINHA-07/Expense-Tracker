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
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

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
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
]


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
# 4. CREATE CHARTS
# ──────────────────────────────────────────────────────────────────────

def create_charts(
    by_category: pd.Series,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Generate and save:
      • A horizontal bar chart  → output_dir/category_chart.png
      • A pie chart             → output_dir/pie_chart.png

    Returns the two file paths.
    """
    if by_category.empty:
        console.print("  [yellow]⚠  No category data — skipping charts.[/yellow]")
        return Path(), Path()

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- spinner while rendering ---
    with console.status(
        "[bold cyan]Rendering charts…[/bold cyan]", spinner="aesthetic"
    ):
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

        cats = by_category.index.tolist()
        vals = by_category.values
        colours = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(cats))]

        # ── Bar Chart ──
        fig_bar, ax_bar = plt.subplots(figsize=(9, 5))
        bars = ax_bar.barh(
            cats[::-1], vals[::-1], color=colours[::-1], height=0.6,
        )
        ax_bar.set_xlabel("Amount (₹)")
        ax_bar.set_title(
            "Spending by Category", fontsize=14, fontweight="bold", pad=12,
        )
        ax_bar.bar_label(bars, fmt="₹{:,.0f}", padding=6, fontsize=9)
        ax_bar.set_xlim(right=float(vals.max()) * 1.20)
        fig_bar.tight_layout()

        bar_path = output_dir / "category_chart.png"
        fig_bar.savefig(bar_path, dpi=150)
        plt.close(fig_bar)

        # ── Pie Chart ──
        fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax_pie.pie(
            vals,
            labels=cats,
            autopct="%1.1f%%",
            startangle=140,
            colors=colours,
            pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        ax_pie.set_title(
            "Expense Breakdown", fontsize=14, fontweight="bold", pad=16,
        )
        fig_pie.tight_layout()

        pie_path = output_dir / "pie_chart.png"
        fig_pie.savefig(pie_path, dpi=150)
        plt.close(fig_pie)

    console.print(
        f"  [bold green]✓[/bold green]  Bar chart  → [link=file://{bar_path.resolve()}]{bar_path}[/link]\n"
        f"  [bold green]✓[/bold green]  Pie chart  → [link=file://{pie_path.resolve()}]{pie_path}[/link]"
    )
    return bar_path, pie_path


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
            create_charts(summary["by_category"], OUTPUT_DIR)
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