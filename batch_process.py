"""
batch_process.py
----------------
Batch pipeline for all Equinix tutorials:
  1. Parse script
  2. Load experimental data
  3. Fit with ALL parameters checkmarked
  4. Export fitted data to Excel
  5. Save snapshot PNG (plot + fitted parameters side by side)

Outputs saved to:  <output_dir>/<tutorial_name>/
  tutorial_N.xlsx   — fitted data export
  tutorial_N.png    — snapshot

Usage:
    cd <folder containing app.py and equinix_*.py>
    python batch_process.py [tutorials_dir] [output_dir]

Defaults:
    tutorials_dir = ./tutorials/
    output_dir    = ./batch_output/
"""

import sys
import os
import io
import time
import traceback
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from equinix_parser import parse_script
from equinix_network import build_network, detect_thermodynamic_cycles
from equinix_kinetics import (build_kinetics_logk_dict, compute_kinetics_curve,
                               fit_kinetics, _collect_all_kinetic_species)
from equinix_kinetics_nmr import (fit_kinetics_nmr_shifts,
                                   fit_kinetics_nmr_integration,
                                   fit_kinetics_nmr_mixed)
from equinix_kinetics_spectra import fit_kinetics_spectra
from equinix_curve import (compute_curve, evaluate_x_expression, _x_per_equiv)
from equinix_fit_conc import fit_parameters
from equinix_fit_nmr import (fit_nmr_shifts, fit_nmr_integration, fit_nmr_mixed)
from equinix_fit_spectra import fit_spectra
from equinix_io import (load_experimental_data, load_spectra_data,
                        export_to_excel, _export_kinetics_excel,
                        generate_parameters_text, generate_kinetics_parameters_text,
                        create_snapshot, text_to_image)

# ── Input-file mapping ────────────────────────────────────────────────────────
INPUT_MAP = {
    "tutorial_1":   "tutorial_1_input",
    "tutorial_1B":  "tutorial_1_input",
    "tutorial_2":   "tutorial_2_input",
    "tutorial_2B":  "tutorial_2_input",
    "tutorial_2C":  "tutorial_2_input",
    "tutorial_3":   "tutorial_3_input",
    "tutorial_4":   "tutorial_4_input",
    "tutorial_5":   "tutorial_5_input",
    "tutorial_6":   "tutorial_6_input",
    "tutorial_7":   "tutorial_7_input",
    "tutorial_8":   "tutorial_8_input",
    "tutorial_9":   "tutorial_9_input",
    "tutorial_10":  "tutorial_10_input",
    "tutorial_11":  "tutorial_11_input",
    "tutorial_12":  "tutorial_12_input",
    "tutorial_12B": "tutorial_12_input",
    "tutorial_13":  "tutorial_13_input",
    "tutorial_14":  "tutorial_14_input",
    "tutorial_15":  "tutorial_15_input",
    "tutorial_16":  "tutorial_16_input",
}

# ── Colour palette (matches the app) ─────────────────────────────────────────
PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_params(parsed, network):
    titrant_key        = network["titrant_key"]
    titrant_name       = network["titrant_name"]
    titrant_free_names = network["titrant_free_names"]
    titrant_keys       = network["titrant_keys"]

    conc_roots = {}
    for cname, cval in parsed["concentrations"].items():
        root = cname[:-1] if cname.endswith("0") else cname
        conc_roots[root] = cval

    primary = list(conc_roots.keys())[0] if conc_roots else network["free_species"][0]
    V0_mL   = list(parsed["volumes"].values())[0] if parsed["volumes"] else 0.5

    titrant_mMs    = {}
    titrant_ratios = {}
    if parsed["titrant_is_solid"]:
        for tkey in titrant_keys:
            tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
            titrant_mMs[tfree]    = 0.0
            titrant_ratios[tfree] = parsed["titrant_ratios"].get(tkey, 1.0)
    else:
        for tkey in titrant_keys:
            tfree = tkey[:-1] if (tkey.endswith("t") or tkey.endswith("0")) else tkey
            titrant_mMs[tfree]    = parsed["titrant"].get(tkey, 10.0)
            titrant_ratios[tfree] = 1.0

    x_expr   = parsed.get("plot_x_expr") or \
               f"{titrant_key}/{list(parsed['concentrations'].keys())[0]}"
    xmax     = float(parsed["plot_xmax"])
    xpe      = _x_per_equiv(x_expr, parsed, conc_roots, float(V0_mL),
                             titrant_free_names, titrant_keys, titrant_mMs,
                             titrant_ratios, parsed["titrant_is_solid"], primary)
    maxEquiv = xmax / max(xpe, 1e-12)

    return {
        "conc0":              conc_roots,
        "V0_mL":              float(V0_mL),
        "titrant_name":       titrant_name,
        "titrant_key":        titrant_key,
        "titrant_free_names": titrant_free_names,
        "titrant_keys":       titrant_keys,
        "titrant_mMs":        titrant_mMs,
        "titrant_ratios":     titrant_ratios,
        "titrant_is_solid":   parsed["titrant_is_solid"],
        "maxEquiv":           maxEquiv,
        "nPts":               200,
        "primary_component":  primary,
    }


def make_eq_figure(curve, x_vals, x_label, plot_y_names, parsed):
    """Build a plotly figure for the equilibrium curve."""
    fig = go.Figure()
    variables = parsed.get("variables", {})
    for i, name in enumerate(plot_y_names):
        color = PALETTE[i % len(PALETTE)]
        if name in curve:
            y = curve[name]
        else:
            # Try as variable
            from equinix_network import compute_variable_curve
            try:
                y = compute_variable_curve(name, variables, curve,
                                           {"all_species": list(curve.keys())}, x_vals)
            except Exception:
                continue
        fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines",
                                 name=name, line=dict(color=color, width=2)))
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title="Concentration (mM)",
        height=500, width=700,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


def make_kin_figure(kin_curve, t_vals, plot_y_names):
    """Build a plotly figure for the kinetics curve."""
    fig = go.Figure()
    for i, name in enumerate(plot_y_names):
        if name not in kin_curve:
            continue
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=t_vals, y=kin_curve[name] * 1000,
                                 mode="lines", name=name,
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Concentration (mM)",
        height=500, width=700,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


# ── Equilibrium pipeline ──────────────────────────────────────────────────────

def process_equilibrium(name, script_text, data_path, out_dir, script_path=None):
    parsed  = parse_script(script_text)
    network = build_network(parsed)
    params  = build_params(parsed, network)

    # Start from script defaults
    logK_vals = {eq["kname"]: eq["logK"] for eq in parsed["equilibria"]}
    # Fit ALL keys
    fit_keys  = list(logK_vals.keys())

    curve = compute_curve(parsed, network, logK_vals, params)
    x_expr = parsed.get("plot_x_expr") or \
             f"{network['titrant_key']}/{list(parsed['concentrations'].keys())[0]}"
    x_vals, x_label = evaluate_x_expression(x_expr, curve, parsed)

    fitted_logKs = logK_vals.copy()
    fit_note     = "no data"

    if data_path and os.path.exists(data_path):
        with open(data_path, "rb") as f:
            file_bytes = f.read()

        nmr_cfg     = parsed.get("nmr")
        spectra_cfg = parsed.get("spectra")
        tol, maxiter = 1e-5, 2000

        if spectra_cfg is not None:
            spectra_data = load_spectra_data(file_bytes)
            wl = spectra_data["wavelengths"]
            success, fitted, stats, msg = fit_spectra(
                parsed, network, spectra_data, params, logK_vals, fit_keys,
                x_expr, float(wl[0]), float(wl[-1]), tol, maxiter, timeout_s=60.0)
        elif nmr_cfg is not None:
            mode     = nmr_cfg["mode"]
            exp_data = load_experimental_data(file_bytes)
            if mode == "shift":
                success, fitted, stats, msg = fit_nmr_shifts(
                    parsed, network, exp_data, params, logK_vals,
                    fit_keys, x_expr, tol, maxiter, timeout_s=60.0)
            elif mode == "integration":
                success, fitted, stats, msg = fit_nmr_integration(
                    parsed, network, exp_data, params, logK_vals,
                    fit_keys, x_expr, tol, maxiter, timeout_s=60.0)
            else:
                success, fitted, stats, msg = fit_nmr_mixed(
                    parsed, network, exp_data, params, logK_vals,
                    fit_keys, x_expr, tol, maxiter, timeout_s=60.0)
        else:
            exp_data = load_experimental_data(file_bytes)
            success, fitted, stats, msg = fit_parameters(
                parsed, network, exp_data, params, logK_vals,
                fit_keys, x_expr, tol, maxiter)

        fitted_logKs = fitted if fitted else logK_vals
        fit_note     = "converged" if success else "no-conv"

        # Recompute curve with fitted values
        curve  = compute_curve(parsed, network, fitted_logKs, params)
        x_vals, x_label = evaluate_x_expression(x_expr, curve, parsed)

    # ── Excel export ──────────────────────────────────────────────────────
    plot_y = parsed["plot_y"] or network["all_species"][:6]
    excel_bytes, _ = export_to_excel(curve, x_vals, parsed, params,
                                     network, script_text, fitted_logKs,
                                     script_path=script_path, input_path=data_path)
    xl_path = os.path.join(out_dir, f"{name}.xlsx")
    with open(xl_path, "wb") as f:
        f.write(excel_bytes)

    # ── Snapshot ──────────────────────────────────────────────────────────
    fig      = make_eq_figure(curve, x_vals, x_label, plot_y, parsed)
    snap, _  = create_snapshot(fig, parsed, params, fitted_logKs)
    png_path = os.path.join(out_dir, f"{name}.png")
    with open(png_path, "wb") as f:
        f.write(snap)

    return fit_note, xl_path, png_path


# ── Kinetics pipeline ─────────────────────────────────────────────────────────

def process_kinetics(name, script_text, data_path, out_dir, script_path=None):
    parsed    = parse_script(script_text)
    logk_dict = build_kinetics_logk_dict(parsed)
    t_max     = float(parsed["plot_xmax"])
    fit_keys  = list(logk_dict.keys())

    kin_curve = compute_kinetics_curve(parsed, logk_dict, t_max, 200)
    t_vals    = kin_curve.get("_t", np.linspace(0, t_max, 200))

    fitted_logk = logk_dict.copy()
    fit_note    = "no data"

    if data_path and os.path.exists(data_path):
        with open(data_path, "rb") as f:
            file_bytes = f.read()

        nmr_cfg     = parsed.get("nmr")
        spectra_cfg = parsed.get("spectra")
        tol, maxiter = 1e-5, 2000

        if spectra_cfg is not None:
            spectra_data = load_spectra_data(file_bytes)
            wl = spectra_data["wavelengths"]
            success, fitted, stats, msg = fit_kinetics_spectra(
                parsed, logk_dict, spectra_data, fit_keys,
                t_max, 200, float(wl[0]), float(wl[-1]), tol, maxiter, timeout_s=60.0)
        elif nmr_cfg is not None:
            mode     = nmr_cfg["mode"]
            nmr_data = load_experimental_data(file_bytes)
            if mode == "shift":
                success, fitted, stats, msg = fit_kinetics_nmr_shifts(
                    parsed, logk_dict, nmr_data, fit_keys,
                    t_max, 200, tol, maxiter, timeout_s=60.0)
            elif mode == "integration":
                success, fitted, stats, msg = fit_kinetics_nmr_integration(
                    parsed, logk_dict, nmr_data, fit_keys,
                    t_max, 200, tol, maxiter, timeout_s=60.0)
            else:
                success, fitted, stats, msg = fit_kinetics_nmr_mixed(
                    parsed, logk_dict, nmr_data, fit_keys,
                    t_max, 200, tol, maxiter, timeout_s=60.0)
        else:
            exp_data = load_experimental_data(file_bytes)
            success, fitted, stats, msg = fit_kinetics(
                parsed, exp_data, logk_dict, fit_keys,
                t_max, 200, tol, maxiter, timeout_s=60.0)

        fitted_logk = fitted if fitted else logk_dict
        fit_note    = "converged" if success else "no-conv"

        # Recompute with fitted values
        kin_curve = compute_kinetics_curve(parsed, fitted_logk, t_max, 200)
        t_vals    = kin_curve.get("_t", np.linspace(0, t_max, 200))

    # ── Excel export ──────────────────────────────────────────────────────
    all_sp    = _collect_all_kinetic_species(parsed)
    plot_y    = parsed["plot_y"] or all_sp[:6]
    variables = parsed.get("variables", {})
    xl_bytes  = _export_kinetics_excel(kin_curve, t_vals, plot_y,
                                       parsed, fitted_logk, script_text, variables,
                                       script_path=script_path, input_path=data_path)
    xl_path   = os.path.join(out_dir, f"{name}.xlsx")
    with open(xl_path, "wb") as f:
        f.write(xl_bytes)

    # ── Snapshot: use generate_kinetics_parameters_text + make figure ────
    params_text = generate_kinetics_parameters_text(parsed, fitted_logk, script_text)
    params_img  = text_to_image(params_text, width=600)

    fig         = make_kin_figure(kin_curve, t_vals, plot_y)
    from PIL import Image
    try:
        plot_bytes = fig.to_image(format="png", width=800, height=600)
        plot_img   = Image.open(io.BytesIO(plot_bytes))
    except Exception:
        plot_img = Image.new("RGB", (800, 600), color="#f0f0f0")
        from PIL import ImageDraw
        d = ImageDraw.Draw(plot_img)
        d.text((400, 300), "(install kaleido for plot image)", fill="#888888", anchor="mm")

    target_h    = max(plot_img.height, params_img.height)
    if plot_img.height != target_h:
        plot_img = plot_img.resize(
            (int(plot_img.width * target_h / plot_img.height), target_h),
            Image.Resampling.LANCZOS)
    if params_img.height != target_h:
        params_img = params_img.resize(
            (int(params_img.width * target_h / params_img.height), target_h),
            Image.Resampling.LANCZOS)

    combined = Image.new("RGB", (plot_img.width + params_img.width, target_h), "white")
    combined.paste(plot_img,   (0, 0))
    combined.paste(params_img, (plot_img.width, 0))

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    png_path = os.path.join(out_dir, f"{name}.png")
    with open(png_path, "wb") as f:
        f.write(buf.getvalue())

    return fit_note, xl_path, png_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tdir     = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
                   os.path.dirname(os.path.abspath(__file__)), "tutorials")
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
                   os.path.dirname(os.path.abspath(__file__)), "batch_output")

    os.makedirs(out_dir, exist_ok=True)

    scripts = sorted(
        f for f in os.listdir(tdir) if f.endswith(".txt") and f.startswith("tutorial_")
    )

    if not scripts:
        print(f"No tutorial_*.txt files found in {tdir}")
        sys.exit(1)

    print(f"\nEquinix batch processing — {len(scripts)} tutorials")
    print(f"  Input:   {tdir}")
    print(f"  Output:  {out_dir}\n")
    print(f"{'Tutorial':<20} {'Fit':<12} {'Time':>6}   Files")
    print("─" * 80)

    passed = failed = 0

    for fname in scripts:
        name        = fname.replace(".txt", "")
        script_path = os.path.join(tdir, fname)
        input_stem  = INPUT_MAP.get(name)
        data_path   = os.path.join(tdir, input_stem + ".xlsx") if input_stem else None
        script_text = open(script_path).read()

        is_kinetics = any(
            (">" in line or "<>" in line) and ";" in line
            for line in script_text.splitlines()
            if not line.strip().startswith("#")
        )
        runner = process_kinetics if is_kinetics else process_equilibrium

        t0 = time.time()
        try:
            fit_note, xl_path, png_path = runner(name, script_text, data_path, out_dir,
                                                   script_path=script_path)
            elapsed = time.time() - t0
            xl_name  = os.path.basename(xl_path)
            png_name = os.path.basename(png_path)
            print(f"  {name:<20} {fit_note:<12} {elapsed:5.1f}s   {xl_name}, {png_name}")
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            tb_lines = traceback.format_exc().strip().split("\n")
            print(f"  {name:<20} {'FAIL':<12} {elapsed:5.1f}s   {e}")
            for line in tb_lines[-3:]:
                print(f"    {line}")
            failed += 1

    print("─" * 80)
    print(f"\n  {passed} succeeded, {failed} failed")
    print(f"  Output folder: {out_dir}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
