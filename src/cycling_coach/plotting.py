"""Shared chart style: validated categorical palette and quiet axes."""

import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#b5b3ad"


def style_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=TEXT_PRIMARY, fontsize=12)


def label(ax, x: str | None = None, y: str | None = None) -> None:
    if x:
        ax.set_xlabel(x, color=TEXT_SECONDARY)
    if y:
        ax.set_ylabel(y, color=TEXT_SECONDARY)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
