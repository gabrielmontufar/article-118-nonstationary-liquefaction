"""Reproducible benchmark for non-stationary liquefaction probability.

The script creates a synthetic but transparent layered soil profile and compares
deterministic, stationary-probabilistic, and non-stationary probabilistic
liquefaction assessments under evolving groundwater and gradation scenarios.
It is intentionally self-contained: only numpy and pandas are required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(1182026)
N_MC = 12000
YEARS = np.arange(0, 51, 2)
GAMMA_TOTAL = 18.2
GAMMA_W = 9.81
MAGNITUDE_SCALING = 1.12


LAYERS = pd.DataFrame(
    [
        {"layer": "L1", "z_mid_m": 2.0, "thickness_m": 2.0, "n160": 10.0, "fc0_pct": 8.0, "d50_mm": 0.33},
        {"layer": "L2", "z_mid_m": 4.5, "thickness_m": 3.0, "n160": 13.0, "fc0_pct": 14.0, "d50_mm": 0.25},
        {"layer": "L3", "z_mid_m": 8.0, "thickness_m": 4.0, "n160": 18.0, "fc0_pct": 21.0, "d50_mm": 0.18},
        {"layer": "L4", "z_mid_m": 12.5, "thickness_m": 5.0, "n160": 24.0, "fc0_pct": 28.0, "d50_mm": 0.14},
    ]
)

SCENARIOS = [
    {"scenario": "stationary", "trend_m_per_yr": 0.0, "season_amp_m": 0.0, "extreme_drop_m": 0.0},
    {"scenario": "rising", "trend_m_per_yr": -0.035, "season_amp_m": 0.0, "extreme_drop_m": 0.0},
    {"scenario": "seasonal", "trend_m_per_yr": -0.010, "season_amp_m": 0.45, "extreme_drop_m": 0.0},
    {"scenario": "extreme", "trend_m_per_yr": -0.025, "season_amp_m": 0.30, "extreme_drop_m": -1.20},
]

GRADATIONS = [
    {"gradation": "constant", "fc_rate_pct_per_yr": 0.00},
    {"gradation": "fines_accumulation", "fc_rate_pct_per_yr": 0.10},
    {"gradation": "fines_washout", "fc_rate_pct_per_yr": -0.08},
]


def rd_youd_2001(z: np.ndarray | float) -> np.ndarray | float:
    """Stress reduction factor approximation for shallow depths."""
    z_arr = np.asarray(z, dtype=float)
    rd = np.where(z_arr <= 9.15, 1.0 - 0.00765 * z_arr, 1.174 - 0.0267 * z_arr)
    return np.clip(rd, 0.55, 1.0)


def groundwater_depth(year: float, sc: dict) -> float:
    base = 3.2
    seasonal = sc["season_amp_m"] * math.sin(2.0 * math.pi * year / 10.0)
    event = sc["extreme_drop_m"] if year >= 30 else 0.0
    return max(0.6, base + sc["trend_m_per_yr"] * year + seasonal + event)


def fines_content(fc0: float, year: float, grad: dict) -> float:
    return float(np.clip(fc0 + grad["fc_rate_pct_per_yr"] * year, 0.0, 45.0))


def clean_sand_equivalent(n160: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Clean-sand equivalent SPT resistance using a smooth fines correction."""
    alpha = np.where(fc <= 5, 0.0, np.where(fc < 35, np.exp(1.76 - 190.0 / (fc**2 + 1e-6)), 5.0))
    beta = np.where(fc <= 5, 1.0, np.where(fc < 35, 0.99 + (fc**1.5) / 1000.0, 1.20))
    return alpha + beta * n160


def crr_from_n1_60cs(n1cs: np.ndarray) -> np.ndarray:
    """Simplified CRR curve fitted for benchmark use, bounded to practical values."""
    x = np.clip(n1cs, 2.0, 32.0)
    crr = 0.048 + 0.0067 * x + 0.00032 * x**2
    return np.clip(crr, 0.05, 0.55)


def csr(pga_g: np.ndarray, z: float, gw_depth: np.ndarray) -> np.ndarray:
    sigma_v0 = GAMMA_TOTAL * z
    below = np.maximum(z - gw_depth, 0.0)
    u = GAMMA_W * below
    sigma_eff = np.maximum(sigma_v0 - u, 8.0)
    return 0.65 * pga_g * (sigma_v0 / sigma_eff) * rd_youd_2001(z)


def pf_from_samples(layer: pd.Series, year: float, sc: dict, grad: dict, mode: str) -> dict:
    gw_mean = groundwater_depth(year if mode == "nonstationary" else 0.0, sc)
    fc_mean = fines_content(layer.fc0_pct, year if mode == "nonstationary" else 0.0, grad)

    n160 = RNG.lognormal(mean=math.log(layer.n160), sigma=0.18, size=N_MC)
    fc = np.clip(RNG.normal(fc_mean, 3.0, size=N_MC), 0.0, 45.0)
    gw = np.clip(RNG.normal(gw_mean, 0.35, size=N_MC), 0.4, 7.0)
    pga = np.clip(RNG.lognormal(mean=math.log(0.28), sigma=0.30, size=N_MC), 0.05, 0.85)
    model_error = RNG.normal(1.0, 0.12, size=N_MC)

    n1cs = clean_sand_equivalent(n160, fc)
    resistance = crr_from_n1_60cs(n1cs) / MAGNITUDE_SCALING
    demand = csr(pga, layer.z_mid_m, gw)
    fs = resistance / np.maximum(demand * model_error, 1e-6)
    pf = float(np.mean(fs < 1.0))
    beta = -float(np.quantile(fs - 1.0, 0.50)) / max(float(np.std(fs - 1.0)), 1e-6)

    det_n1cs = clean_sand_equivalent(np.array([layer.n160]), np.array([fc_mean]))[0]
    det_crr = crr_from_n1_60cs(np.array([det_n1cs]))[0] / MAGNITUDE_SCALING
    det_csr = csr(np.array([0.28]), layer.z_mid_m, np.array([gw_mean]))[0]
    fs_det = float(det_crr / det_csr)

    return {
        "pf": pf,
        "beta_proxy": beta,
        "fs_deterministic": fs_det,
        "gw_depth_m": gw_mean,
        "fc_pct": fc_mean,
        "n1_60cs": float(det_n1cs),
        "crr": float(det_crr),
        "csr": float(det_csr),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sc in SCENARIOS:
        for grad in GRADATIONS:
            for _, layer in LAYERS.iterrows():
                for year in YEARS:
                    stationary = pf_from_samples(layer, year, sc, grad, "stationary")
                    nonstationary = pf_from_samples(layer, year, sc, grad, "nonstationary")
                    se_stat = math.sqrt(max(stationary["pf"] * (1.0 - stationary["pf"]), 0.0) / N_MC)
                    se_non = math.sqrt(max(nonstationary["pf"] * (1.0 - nonstationary["pf"]), 0.0) / N_MC)
                    rows.append(
                        {
                            "scenario": sc["scenario"],
                            "gradation": grad["gradation"],
                            "layer": layer.layer,
                            "z_mid_m": layer.z_mid_m,
                            "year": year,
                            "pf_stationary": stationary["pf"],
                            "pf_stationary_ci_low": max(0.0, stationary["pf"] - 1.96 * se_stat),
                            "pf_stationary_ci_high": min(1.0, stationary["pf"] + 1.96 * se_stat),
                            "pf_nonstationary": nonstationary["pf"],
                            "pf_nonstationary_ci_low": max(0.0, nonstationary["pf"] - 1.96 * se_non),
                            "pf_nonstationary_ci_high": min(1.0, nonstationary["pf"] + 1.96 * se_non),
                            "delta_pf": nonstationary["pf"] - stationary["pf"],
                            "fs_deterministic_nonstationary": nonstationary["fs_deterministic"],
                            "gw_depth_m": nonstationary["gw_depth_m"],
                            "fc_pct": nonstationary["fc_pct"],
                            "n1_60cs": nonstationary["n1_60cs"],
                            "crr": nonstationary["crr"],
                            "csr": nonstationary["csr"],
                            "beta_proxy": nonstationary["beta_proxy"],
                        }
                    )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["scenario", "gradation"], as_index=False)
        .agg(
            max_pf_nonstationary=("pf_nonstationary", "max"),
            mean_pf_nonstationary=("pf_nonstationary", "mean"),
            max_delta_pf=("delta_pf", "max"),
            final_mean_pf=("pf_nonstationary", lambda s: float(s[results.loc[s.index, "year"].eq(YEARS[-1])].mean())),
            min_fs_deterministic=("fs_deterministic_nonstationary", "min"),
        )
        .sort_values(["max_pf_nonstationary", "max_delta_pf"], ascending=False)
    )
    return results, summary


def svg_line(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]], ylabel: str) -> None:
    width, height = 900, 560
    ml, mr, mt, mb = 90, 30, 70, 70
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = 0.0, max(float(ys.max()) * 1.12, 0.05)

    def sx(x): return ml + (x - xmin) / (xmax - xmin) * (width - ml - mr)
    def sy(y): return height - mb - (y - ymin) / (ymax - ymin) * (height - mt - mb)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{title}</text>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Time horizon (years)</text>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>',
    ]
    for tick in np.linspace(xmin, xmax, 6):
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{height-mb}" x2="{x:.1f}" y2="{height-mb+6}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-mb+24}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.0f}</text>')
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        parts.append(f'<line x1="{ml-6}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{ml-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for i, (label, xdata, ydata) in enumerate(series):
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xdata, ydata))
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<rect x="{width-260}" y="{70+i*24}" width="16" height="3" fill="{color}"/>')
        parts.append(f'<text x="{width-238}" y="{76+i*24}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, data: pd.DataFrame, title: str) -> None:
    width, height = 820, 560
    ml, mr, mt, mb = 100, 110, 70, 70
    years = sorted(data.year.unique())
    layers = list(LAYERS.layer)
    val = {(r.layer, r.year): r.pf_nonstationary for r in data.itertuples()}
    maxv = max(val.values()) if val else 1.0
    cw = (width - ml - mr) / len(years)
    ch = (height - mt - mb) / len(layers)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{title}</text>',
    ]
    for i, layer in enumerate(layers):
        y = mt + i * ch
        parts.append(f'<text x="{ml-14}" y="{y+ch/2+5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{layer}</text>')
        for j, year in enumerate(years):
            x = ml + j * cw
            v = val.get((layer, year), 0.0)
            r = int(245 - 70 * (v / maxv))
            g = int(248 - 190 * (v / maxv))
            b = int(255 - 220 * (v / maxv))
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw+0.5:.1f}" height="{ch+0.5:.1f}" fill="rgb({r},{g},{b})" stroke="white"/>')
    for j, year in enumerate(years[::5]):
        x = ml + years.index(year) * cw + cw / 2
        parts.append(f'<text x="{x:.1f}" y="{height-mb+22}" text-anchor="middle" font-family="Arial" font-size="12">{year}</text>')
    parts.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Time horizon (years)</text>')
    parts.append(f'<text x="24" y="{height/2}" transform="rotate(-90 24,{height/2})" text-anchor="middle" font-family="Arial" font-size="14">Layer</text>')
    for k in range(6):
        v = k / 5 * maxv
        y = mt + (5-k) * 50
        r = int(245 - 70 * (v / maxv))
        g = int(248 - 190 * (v / maxv))
        b = int(255 - 220 * (v / maxv))
        parts.append(f'<rect x="{width-mr+35}" y="{y}" width="28" height="50" fill="rgb({r},{g},{b})"/>')
        parts.append(f'<text x="{width-mr+70}" y="{y+30}" font-family="Arial" font-size="12">{v:.2f}</text>')
    parts.append(f'<text x="{width-mr+35}" y="{mt-14}" font-family="Arial" font-size="12">Pf</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def png_line(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]], ylabel: str) -> None:
    width, height = 1400, 850
    ml, mr, mt, mb = 140, 70, 110, 110
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, tick_font = _font(30, True), _font(22), _font(18)
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = 0.0, max(float(ys.max()) * 1.12, 0.05)

    def sx(x): return ml + (x - xmin) / (xmax - xmin) * (width - ml - mr)
    def sy(y): return height - mb - (y - ymin) / (ymax - ymin) * (height - mt - mb)

    draw.text((width / 2, 42), title, anchor="mm", font=title_font, fill=(0, 0, 0))
    draw.line((ml, mt, ml, height - mb, width - mr, height - mb), fill=(40, 40, 40), width=2)
    for tick in np.linspace(xmin, xmax, 6):
        x = sx(tick)
        draw.line((x, height - mb, x, height - mb + 10), fill=(40, 40, 40), width=2)
        draw.text((x, height - mb + 32), f"{tick:.0f}", anchor="mm", font=tick_font, fill=(40, 40, 40))
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        draw.line((ml - 10, y, ml, y), fill=(40, 40, 40), width=2)
        draw.line((ml, y, width - mr, y), fill=(225, 225, 225), width=1)
        draw.text((ml - 18, y), f"{tick:.2f}", anchor="rm", font=tick_font, fill=(40, 40, 40))
    for i, (label, xdata, ydata) in enumerate(series):
        pts = [(sx(float(x)), sy(float(y))) for x, y in zip(xdata, ydata)]
        draw.line(pts, fill=colors[i % len(colors)], width=5)
        yleg = 120 + i * 34
        draw.line((width - 360, yleg, width - 320, yleg), fill=colors[i % len(colors)], width=6)
        draw.text((width - 308, yleg), label, anchor="lm", font=tick_font, fill=(20, 20, 20))
    draw.text((width / 2, height - 34), "Time horizon (years)", anchor="mm", font=label_font, fill=(0, 0, 0))
    rotated = Image.new("RGBA", (420, 50), (255, 255, 255, 0))
    rdraw = ImageDraw.Draw(rotated)
    rdraw.text((210, 25), ylabel, anchor="mm", font=label_font, fill=(0, 0, 0))
    img.paste(rotated.rotate(90, expand=True), (26, int(height / 2 - 210)), rotated.rotate(90, expand=True))
    img.save(path, "PNG")


def png_heatmap(path: Path, data: pd.DataFrame, title: str) -> None:
    width, height = 1300, 820
    ml, mr, mt, mb = 150, 170, 110, 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, tick_font = _font(30, True), _font(22), _font(18)
    years = sorted(data.year.unique())
    layers = list(LAYERS.layer)
    val = {(r.layer, r.year): r.pf_nonstationary for r in data.itertuples()}
    maxv = max(val.values()) if val else 1.0
    cw = (width - ml - mr) / len(years)
    ch = (height - mt - mb) / len(layers)
    draw.text((width / 2, 42), title, anchor="mm", font=title_font, fill=(0, 0, 0))
    for i, layer in enumerate(layers):
        y = mt + i * ch
        draw.text((ml - 18, y + ch / 2), layer, anchor="rm", font=label_font, fill=(20, 20, 20))
        for j, year in enumerate(years):
            x = ml + j * cw
            v = val.get((layer, year), 0.0)
            r = int(245 - 70 * (v / maxv))
            g = int(248 - 190 * (v / maxv))
            b = int(255 - 220 * (v / maxv))
            draw.rectangle((x, y, x + cw + 1, y + ch + 1), fill=(r, g, b), outline="white")
    for year in years[::5]:
        x = ml + years.index(year) * cw + cw / 2
        draw.text((x, height - mb + 34), str(year), anchor="mm", font=tick_font, fill=(40, 40, 40))
    draw.text((width / 2, height - 34), "Time horizon (years)", anchor="mm", font=label_font, fill=(0, 0, 0))
    for k in range(6):
        v = k / 5 * maxv
        y = mt + (5 - k) * 72
        r = int(245 - 70 * (v / maxv))
        g = int(248 - 190 * (v / maxv))
        b = int(255 - 220 * (v / maxv))
        draw.rectangle((width - mr + 45, y, width - mr + 85, y + 72), fill=(r, g, b))
        draw.text((width - mr + 100, y + 36), f"{v:.2f}", anchor="lm", font=tick_font, fill=(40, 40, 40))
    draw.text((width - mr + 45, mt - 24), "Pf", font=tick_font, fill=(40, 40, 40))
    img.save(path, "PNG")


def main() -> None:
    results, summary = run()
    results.to_csv(DATA / "liquefaction_benchmark_results.csv", index=False)
    summary.to_csv(DATA / "liquefaction_benchmark_summary.csv", index=False)
    LAYERS.to_csv(DATA / "synthetic_layer_profile.csv", index=False)

    fig_data = results[(results.scenario == "extreme") & (results.gradation == "fines_accumulation")]
    series = []
    for layer in LAYERS.layer:
        s = fig_data[fig_data.layer == layer].sort_values("year")
        series.append((layer, s.year.to_numpy(), s.pf_nonstationary.to_numpy()))
    svg_line(FIGURES / "fig01_pf_time_extreme_accumulation.svg", "Non-stationary liquefaction probability", series, "Probability of liquefaction")
    png_line(FIGURES / "fig01_pf_time_extreme_accumulation.png", "Non-stationary liquefaction probability", series, "Probability of liquefaction")

    sc_series = []
    for scenario in ["stationary", "rising", "seasonal", "extreme"]:
        s = results[(results.scenario == scenario) & (results.gradation == "fines_accumulation")]
        s = s.groupby("year", as_index=False).pf_nonstationary.mean()
        sc_series.append((scenario, s.year.to_numpy(), s.pf_nonstationary.to_numpy()))
    svg_line(FIGURES / "fig02_profile_mean_pf_by_scenario.svg", "Profile-average probability by groundwater scenario", sc_series, "Mean profile Pf")
    png_line(FIGURES / "fig02_profile_mean_pf_by_scenario.png", "Profile-average probability by groundwater scenario", sc_series, "Mean profile Pf")

    svg_heatmap(FIGURES / "fig03_depth_time_pf_heatmap.svg", fig_data, "Depth-time probability map, extreme + fines accumulation")
    png_heatmap(FIGURES / "fig03_depth_time_pf_heatmap.png", fig_data, "Depth-time probability map, extreme + fines accumulation")

    comp = results.groupby(["scenario", "gradation", "year"], as_index=False).agg(
        stationary=("pf_stationary", "mean"),
        nonstationary=("pf_nonstationary", "mean"),
    )
    comp["delta"] = comp.nonstationary - comp.stationary
    comp.to_csv(DATA / "profile_method_comparison.csv", index=False)
    convergence_rows = []
    target_layer = LAYERS.iloc[0]
    target_sc = next(s for s in SCENARIOS if s["scenario"] == "extreme")
    target_grad = next(g for g in GRADATIONS if g["gradation"] == "fines_washout")
    global N_MC, RNG
    original_n = N_MC
    for n in [1000, 3000, 6000, 12000]:
        N_MC = n
        vals = []
        for rep in range(5):
            RNG = np.random.default_rng(1182026 + n + rep)
            vals.append(pf_from_samples(target_layer, 50, target_sc, target_grad, "nonstationary")["pf"])
        convergence_rows.append(
            {
                "sample_size": n,
                "replicates": 5,
                "mean_pf": float(np.mean(vals)),
                "std_pf": float(np.std(vals, ddof=1)),
                "min_pf": float(np.min(vals)),
                "max_pf": float(np.max(vals)),
            }
        )
    N_MC = original_n
    RNG = np.random.default_rng(1182026)
    pd.DataFrame(convergence_rows).to_csv(DATA / "monte_carlo_convergence_check.csv", index=False)
    print("Wrote benchmark outputs to", ROOT)
    print(summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
