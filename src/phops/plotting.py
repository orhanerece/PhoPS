"""Plotting and image export helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from astropy.stats import mad_std
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
import matplotlib

MPL_DIR = Path(tempfile.gettempdir()) / "phops-mpl"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import AppConfig
from .reporting import ProgressReporter, report


@dataclass
class LightCurvePlotResult:
    """Generated light-curve plot paths."""

    png_path: Path
    pdf_path: Path | None = None


IDL_LIKE_RC = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.05,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
}

LIGHT_CURVE_AUX_META = {
    "snr": {"label": "S/N", "color": "#445d73"},
    "fwhm": {"label": "FWHM (px)", "color": "#5e7280"},
    "zp_scatter": {"label": "ZP scatter (mag)", "color": "#6d7d8c"},
}


def _light_curve_x_values(df: pd.DataFrame, x_axis: str) -> tuple[np.ndarray, str, float]:
    jd_values = np.asarray(df["jd"], dtype=float)
    reference_jd = float(np.nanmin(jd_values))
    if x_axis == "jd":
        return jd_values, "Julian Date (UTC mid-exposure)", reference_jd
    return (jd_values - reference_jd) * 86400.0, f"Time Since Start (s) | JD0={reference_jd:.6f}", reference_jd


def _convert_event_window(
    event_window: tuple[float, float] | None,
    event_unit: str,
    x_axis: str,
    reference_jd: float,
) -> tuple[float, float] | None:
    if event_window is None:
        return None
    start, end = event_window
    if event_unit == x_axis:
        return start, end
    if event_unit == "jd" and x_axis == "relative_seconds":
        return (start - reference_jd) * 86400.0, (end - reference_jd) * 86400.0
    if event_unit == "relative_seconds" and x_axis == "jd":
        return reference_jd + (start / 86400.0), reference_jd + (end / 86400.0)
    return event_window


def _flag_light_curve_outliers(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=bool)

    flags = np.zeros(len(df), dtype=bool)

    if "mag_calib" in df and len(df) >= 7:
        window = min(len(df) if len(df) % 2 == 1 else len(df) - 1, 11)
        window = max(window, 3)
        rolling = df["mag_calib"].rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
        baseline = rolling.fillna(df["mag_calib"].median())
        residual = np.asarray(df["mag_calib"] - baseline, dtype=float)
        scale = float(mad_std(residual, ignore_nan=True))
        if np.isfinite(scale) and scale > 0:
            flags |= np.abs(residual) > (5.0 * scale)

    for column in ("mag_err", "zp_scatter", "fwhm"):
        if column not in df:
            continue
        values = np.asarray(df[column], dtype=float)
        valid = np.isfinite(values)
        if np.sum(valid) < 5:
            continue
        center = float(np.nanmedian(values[valid]))
        scale = float(mad_std(values[valid], ignore_nan=True))
        if np.isfinite(scale) and scale > 0:
            flags |= valid & (values > (center + 4.0 * scale))

    if "snr" in df:
        values = np.asarray(df["snr"], dtype=float)
        valid = np.isfinite(values)
        if np.sum(valid) >= 5:
            center = float(np.nanmedian(values[valid]))
            scale = float(mad_std(values[valid], ignore_nan=True))
            if np.isfinite(scale) and scale > 0:
                flags |= valid & (values < max(center - 4.0 * scale, 0.0))

    return flags


def _light_curve_y_limits(mag_values: np.ndarray, outlier_mask: np.ndarray | None = None) -> tuple[float, float] | None:
    """Return a 3-sigma y-range for the calibrated magnitudes."""

    finite_mask = np.isfinite(mag_values)
    if outlier_mask is not None and len(outlier_mask) == len(mag_values):
        candidate_mask = finite_mask & ~outlier_mask
        if np.sum(candidate_mask) >= 3:
            finite_mask = candidate_mask

    values = np.asarray(mag_values[finite_mask], dtype=float)
    if values.size == 0:
        return None

    _, center, scatter = sigma_clipped_stats(values, sigma=3.0)
    if not (np.isfinite(center) and np.isfinite(scatter)) or scatter <= 0:
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if not (np.isfinite(lower) and np.isfinite(upper)):
            return None
        if lower == upper:
            padding = max(abs(lower) * 0.01, 0.01)
            return lower - padding, upper + padding
        return lower, upper

    lower = float(center - (3.0 * scatter))
    upper = float(center + (3.0 * scatter))
    if lower == upper:
        padding = max(abs(lower) * 0.01, 0.01)
        return lower - padding, upper + padding
    return lower, upper


def _apply_scientific_axis_style(axis) -> None:
    axis.minorticks_on()
    axis.tick_params(which="major", direction="in", top=True, right=True, length=6, width=0.95)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#1a1a1a")
    axis.grid(True, which="major", color="#9aa5b1", alpha=0.2, linestyle=":", linewidth=0.6)
    axis.grid(True, which="minor", color="#c8d0d8", alpha=0.12, linestyle=":", linewidth=0.45)


def _light_curve_stats_lines(
    df: pd.DataFrame,
    mag_values: np.ndarray,
    outlier_mask: np.ndarray,
) -> list[str]:
    valid_mask = np.isfinite(mag_values) & ~outlier_mask
    if np.sum(valid_mask) < 3:
        valid_mask = np.isfinite(mag_values)

    stats_lines = [f"N={int(np.sum(valid_mask))}"]
    if np.any(valid_mask):
        _, center, scatter = sigma_clipped_stats(mag_values[valid_mask], sigma=3.0)
        if np.isfinite(center):
            stats_lines.append(f"median={center:.4f} mag")
        if np.isfinite(scatter) and scatter > 0:
            stats_lines.append(f"sigma={scatter:.4f} mag")

    if "mag_err" in df:
        mag_err = np.asarray(df["mag_err"], dtype=float)
        finite = np.isfinite(mag_err) & valid_mask
        if np.any(finite):
            stats_lines.append(f"med err={np.nanmedian(mag_err[finite]):.4f} mag")

    if "snr" in df:
        snr = np.asarray(df["snr"], dtype=float)
        finite = np.isfinite(snr)
        if np.any(finite):
            stats_lines.append(f"med S/N={np.nanmedian(snr[finite]):.1f}")
    return stats_lines


def plot_astrometry_residuals(
    csv_path: Path,
    output_path: Path,
    telescope_name: str = "PhoPS",
    reporter: ProgressReporter | None = None,
) -> None:
    """Create a summary plot for astrometric residuals."""

    if not csv_path.exists():
        report(reporter, "warning", f"Astrometry CSV not found: {csv_path}", stage="plot")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        report(reporter, "warning", f"Astrometry CSV is empty: {csv_path}", stage="plot")
        return

    dra = df["dra_corr"].values - np.median(df["dra_corr"])
    ddec = df["ddec"].values - np.median(df["ddec"])
    _, _, std_ra = sigma_clipped_stats(dra, sigma=3.0)
    _, _, std_dec = sigma_clipped_stats(ddec, sigma=3.0)
    total_rms = float(np.sqrt(std_ra**2 + std_dec**2))

    sns.set_theme(style="ticks")
    grid = sns.JointGrid(x=dra, y=ddec, space=0)
    grid.plot_joint(sns.scatterplot, s=4, alpha=0.25, color="navy", edgecolor=None, rasterized=True)
    grid.plot_marginals(sns.histplot, kde=True, color="navy", alpha=0.3, bins=60)
    grid.ax_joint.axhline(0, color="black", lw=0.8, ls="--")
    grid.ax_joint.axvline(0, color="black", lw=0.8, ls="--")
    grid.ax_joint.set_xlim(-0.9, 0.9)
    grid.ax_joint.set_ylim(-0.9, 0.9)
    grid.set_axis_labels(r"$\Delta \alpha \cos \delta$ (arcsec)", r"$\Delta \delta$ (arcsec)", fontsize=11)
    info_text = f"{telescope_name} residuals\nRMS: {total_rms:.3f} arcsec\nStars: {len(df)}"
    proxy = mlines.Line2D([], [], color="navy", marker="o", linestyle="None", markersize=5, label=info_text)
    grid.ax_joint.legend(handles=[proxy], loc="upper right", frameon=True, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.subplots_adjust(top=0.92, right=0.92)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    report(reporter, "info", f"Astrometry plot saved: {output_path.name}", stage="plot")


def plot_photometry_light_curve(
    csv_path: Path,
    output_png: Path,
    config: AppConfig,
    output_pdf: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> LightCurvePlotResult | None:
    """Create a light-curve plot with optional quality-control panels."""

    if not csv_path.exists():
        report(reporter, "warning", f"Photometry CSV not found: {csv_path}", stage="plot")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        report(reporter, "warning", f"Photometry CSV is empty: {csv_path}", stage="plot")
        return None

    required_columns = {"jd", "mag_calib", "mag_err"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        report(reporter, "warning", f"Photometry CSV is missing required columns: {', '.join(missing)}", stage="plot")
        return None

    df = df.sort_values("jd").reset_index(drop=True)
    x_values, x_label, reference_jd = _light_curve_x_values(df, config.plots.light_curve_x_axis)
    event_window = _convert_event_window(
        config.plots.light_curve_event_window,
        config.plots.light_curve_event_unit,
        config.plots.light_curve_x_axis,
        reference_jd,
    )
    outlier_mask = _flag_light_curve_outliers(df)

    aux_columns = [column for column in ("snr", "fwhm", "zp_scatter") if column in df.columns]
    show_aux = config.plots.light_curve_aux_panels and bool(aux_columns)
    panel_rows = 1 + len(aux_columns) if show_aux else 1
    height_ratios = [3.4] + [1.0] * len(aux_columns) if show_aux else [1.0]
    with plt.rc_context(IDL_LIKE_RC):
        fig, axes = plt.subplots(
            panel_rows,
            1,
            figsize=(11.5, 4.9 + (1.45 * len(aux_columns))),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        main_ax = axes[0]
        mag_values = np.asarray(df["mag_calib"], dtype=float)
        mag_errors = np.asarray(df["mag_err"], dtype=float)
        normal_mask = ~outlier_mask

        if event_window is not None:
            for axis in axes:
                axis.axvspan(event_window[0], event_window[1], color="#cab27b", alpha=0.16, lw=0, zorder=0)

        if np.sum(normal_mask) > 1:
            main_ax.plot(
                x_values[normal_mask],
                mag_values[normal_mask],
                color="#9eb2c6",
                lw=0.9,
                alpha=0.8,
                zorder=1,
            )

        valid_mag_mask = np.isfinite(mag_values) & normal_mask
        if np.any(valid_mag_mask):
            _, median_mag, clipped_scatter = sigma_clipped_stats(mag_values[valid_mag_mask], sigma=3.0)
            if np.isfinite(median_mag):
                main_ax.axhline(median_mag, color="#556270", lw=0.95, ls="--", alpha=0.8, zorder=1)
            if np.isfinite(clipped_scatter) and clipped_scatter > 0:
                main_ax.axhspan(
                    median_mag - clipped_scatter,
                    median_mag + clipped_scatter,
                    color="#dbe4ec",
                    alpha=0.35,
                    zorder=0,
                )

        main_ax.errorbar(
            x_values[normal_mask],
            mag_values[normal_mask],
            yerr=mag_errors[normal_mask],
            fmt="o",
            ms=4.6,
            mfc="#102a43",
            mec="white",
            mew=0.45,
            color="#102a43",
            ecolor="#7f94a8",
            elinewidth=0.85,
            capsize=0,
            alpha=0.96,
            label="Measurements",
            zorder=3,
        )
        if np.any(outlier_mask):
            main_ax.errorbar(
                x_values[outlier_mask],
                mag_values[outlier_mask],
                yerr=mag_errors[outlier_mask],
                fmt="D",
                ms=5.0,
                mfc="white",
                mec="#b54708",
                mew=1.0,
                color="#b54708",
                ecolor="#b54708",
                elinewidth=0.8,
                capsize=0,
                alpha=0.95,
                label="Flagged outliers",
                zorder=4,
            )

        y_limits = _light_curve_y_limits(mag_values, outlier_mask=outlier_mask)
        if y_limits is not None:
            main_ax.set_ylim(y_limits[0], y_limits[1])
        main_ax.invert_yaxis()
        main_ax.set_ylabel(f"Calibrated Magnitude ({config.photometry.filter})", labelpad=10)

        if config.photometry.mode == "asteroid":
            title_target = config.photometry.target_id or "asteroid"
        else:
            title_target = "fixed-star target"
        main_ax.set_title(
            f"{title_target} | {config.photometry.filter}-band calibrated light curve",
            loc="left",
            pad=10,
        )
        subtitle = f"{config.paths.photometry_csv_path.name} | {len(df)} frames | x-axis={config.plots.light_curve_x_axis}"
        main_ax.text(0.0, 1.015, subtitle, transform=main_ax.transAxes, fontsize=9, color="#52606d")
        if event_window is not None:
            main_ax.text(
                0.995,
                1.015,
                "event window",
                transform=main_ax.transAxes,
                ha="right",
                fontsize=9,
                color="#8d6e1f",
            )
        stats_text = "\n".join(_light_curve_stats_lines(df, mag_values, outlier_mask))
        main_ax.text(
            0.985,
            0.04,
            stats_text,
            transform=main_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#243b53",
            bbox={"facecolor": "white", "edgecolor": "#d2d8de", "boxstyle": "round,pad=0.28", "alpha": 0.92},
        )
        main_ax.legend(loc="upper left", fontsize=9)
        main_ax.margins(x=0.015)
        _apply_scientific_axis_style(main_ax)

        for axis, column in zip(axes[1:], aux_columns):
            values = np.asarray(df[column], dtype=float)
            meta = LIGHT_CURVE_AUX_META.get(column, {"label": column.replace("_", " "), "color": "#5b6770"})
            axis.plot(x_values, values, color=meta["color"], lw=1.0, alpha=0.85, zorder=1)
            axis.scatter(
                x_values[normal_mask],
                values[normal_mask],
                s=14,
                color=meta["color"],
                alpha=0.78,
                zorder=3,
            )
            if np.any(outlier_mask):
                axis.scatter(
                    x_values[outlier_mask],
                    values[outlier_mask],
                    s=22,
                    marker="D",
                    facecolors="white",
                    edgecolors="#b54708",
                    linewidths=1.0,
                    zorder=4,
                )
            finite_values = np.isfinite(values)
            if np.any(finite_values):
                axis.axhline(np.nanmedian(values[finite_values]), color="#8a98a6", lw=0.8, ls="--", alpha=0.7, zorder=0)
            axis.set_ylabel(meta["label"], labelpad=10)
            axis.margins(x=0.015)
            _apply_scientific_axis_style(axis)

        axes[-1].set_xlabel(x_label, labelpad=8)
        fig.align_ylabels(axes)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=320, bbox_inches="tight")
        if output_pdf is not None:
            fig.savefig(output_pdf, dpi=320, bbox_inches="tight")
        plt.close(fig)

    report(reporter, "info", f"Light-curve plot saved: {output_png.name}", stage="plot")
    if output_pdf is not None:
        report(reporter, "info", f"Light-curve PDF saved: {output_pdf.name}", stage="plot")
    return LightCurvePlotResult(png_path=output_png, pdf_path=output_pdf)


def plot_photometry_sources(
    data: np.ndarray,
    image_sources,
    matched_sources,
    wcs,
    output_path: Path,
    mode: str,
    reporter: ProgressReporter | None = None,
    target_coord: tuple[float, float] | None = None,
) -> None:
    """Plot detected and matched sources in either pixel or WCS space."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = data.shape

    if mode == "wcs":
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=wcs)
        norm = simple_norm(data, stretch="sinh", percent=97.0, min_percent=0.1)
        ax.imshow(data, cmap="viridis", origin="lower", norm=norm)
        transform = ax.get_transform("pixel")
        ax.scatter(image_sources["xcentroid"], image_sources["ycentroid"], transform=transform, edgecolor="cyan", facecolor="none", s=50, alpha=0.7)
        ax.scatter(matched_sources["xcentroid"], matched_sources["ycentroid"], transform=transform, edgecolor="red", facecolor="none", s=90, lw=1.3)
        if target_coord is not None:
            ax.scatter([target_coord[0]], [target_coord[1]], transform=transform, edgecolor="white", facecolor="none", s=160, lw=2.0)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.coords[0].set_axislabel("Right Ascension")
        ax.coords[1].set_axislabel("Declination")
        ax.grid(color="white", alpha=0.4, linewidth=0.5, linestyle="--")
        ax.legend(
            [
                mlines.Line2D([], [], color="cyan", marker="o", linestyle="None", markerfacecolor="none"),
                mlines.Line2D([], [], color="red", marker="o", linestyle="None", markerfacecolor="none"),
            ],
            [f"Detected ({len(image_sources)})", f"Matched ({len(matched_sources)})"],
            loc="upper left",
            framealpha=0.5,
        )
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        vmin, vmax = np.percentile(data, [30, 99])
        ax.imshow(data, cmap="gray_r", origin="lower", vmin=vmin, vmax=vmax)
        ax.scatter(image_sources["xcentroid"], image_sources["ycentroid"], edgecolor="blue", facecolor="none", s=18, alpha=0.8)
        ax.scatter(matched_sources["xcentroid"], matched_sources["ycentroid"], edgecolor="red", facecolor="none", s=90, lw=1.5)
        if target_coord is not None:
            ax.scatter([target_coord[0]], [target_coord[1]], edgecolor="green", facecolor="none", s=180, lw=2.0)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        ax.legend(
            [
                mlines.Line2D([], [], color="blue", marker="o", linestyle="None", markerfacecolor="none"),
                mlines.Line2D([], [], color="red", marker="o", linestyle="None", markerfacecolor="none"),
            ],
            [f"Detected ({len(image_sources)})", f"Matched ({len(matched_sources)})"],
            loc="upper right",
            framealpha=0.5,
        )
        plt.savefig(output_path, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)

    report(reporter, "info", f"Source plot saved: {output_path.name}", stage="plot")


def save_target_cutout(
    data: np.ndarray,
    x: float,
    y: float,
    filename: str,
    output_dir: Path,
    reporter: ProgressReporter | None = None,
) -> Path:
    """Save a cutout around the measured target."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cutout_{filename}.png"
    cutout = Cutout2D(data, (x, y), (200, 200), mode="partial", fill_value=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    vmin, vmax = np.percentile(cutout.data, [50, 99])
    ax.imshow(cutout.data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    cx, cy = cutout.input_position_cutout
    circle = plt.Circle((cx, cy), radius=5, color="cyan", fill=False, lw=1.5)
    ax.add_patch(circle)
    ax.set_title(filename, fontsize=8)
    ax.axis("off")
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    report(reporter, "info", f"Target cutout saved: {output_path.name}", stage="plot")
    return output_path
