"""
run_tests.py
------------
Headless regression tests for all Equinix tutorials.
Runs the full backend pipeline (parse → network/ODE → curve → fit)
without Streamlit. Tests every tutorial script + data file.

Usage:
    cd <folder containing app.py and equinix_*.py>
    python run_tests.py [path/to/tutorials/]

If no path is given, looks for tutorials in ./tutorials/
"""

import sys
import os
import time
import traceback
import numpy as np

# ── Allow running from the equinix folder ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from equinix_parser import parse_script
from equinix_network import build_network
from equinix_kinetics import (build_kinetics_logk_dict, compute_kinetics_curve,
                               fit_kinetics, _collect_all_kinetic_species)
from equinix_kinetics_nmr import (fit_kinetics_nmr_shifts,
                                   fit_kinetics_nmr_integration,
                                   fit_kinetics_nmr_mixed)
from equinix_kinetics_spectra import fit_kinetics_spectra
from equinix_curve import (compute_curve, evaluate_x_expression,
                            _x_per_equiv, find_equiv_for_x)
from equinix_fit_conc import fit_parameters
from equinix_fit_nmr import (fit_nmr_shifts, fit_nmr_integration, fit_nmr_mixed,
                              compute_nmr_curves)
from equinix_fit_spectra import fit_spectra
from equinix_io import load_experimental_data, load_spectra_data


# ── Input-file mapping  (tutorial name → data file stem) ────────────────────
# Variants (1B, 1C, 2B …) share the base tutorial's input file.
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_params(parsed, network):
    """Build a params dict from parsed script defaults (mirrors what the UI does)."""
    titrant_key        = network["titrant_key"]
    titrant_name       = network["titrant_name"]
    titrant_free_names = network["titrant_free_names"]
    titrant_keys       = network["titrant_keys"]

    conc_roots = {}
    for cname, cval in parsed["concentrations"].items():
        root = cname[:-1] if cname.endswith("0") else cname
        conc_roots[root] = cval

    primary_component = list(conc_roots.keys())[0] if conc_roots else network["free_species"][0]

    V0_mL = list(parsed["volumes"].values())[0] if parsed["volumes"] else 0.5

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

    xmax = float(parsed["plot_xmax"])
    x_expr = parsed.get("plot_x_expr") or f"{titrant_key}/{list(parsed['concentrations'].keys())[0]}"

    xpe = _x_per_equiv(
        x_expr, parsed, conc_roots, float(V0_mL),
        titrant_free_names, titrant_keys, titrant_mMs, titrant_ratios,
        parsed["titrant_is_solid"], primary_component,
    )
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
        "nPts":               100,
        "primary_component":  primary_component,
    }


def run_equilibrium(name, script_text, data_path, tdir):
    """Full equilibrium pipeline: parse → network → curve → fit (if data)."""
    steps = []

    # 1. Parse
    parsed = parse_script(script_text)
    steps.append("parse")

    # 2. Build network
    network = build_network(parsed)
    steps.append("build_network")

    # 3. Params + curve
    params   = build_params(parsed, network)
    logK_vals = {eq["kname"]: eq["logK"] for eq in parsed["equilibria"]}
    curve    = compute_curve(parsed, network, logK_vals, params)
    x_expr   = parsed.get("plot_x_expr") or \
               f"{network['titrant_key']}/{list(parsed['concentrations'].keys())[0]}"
    x_vals, _ = evaluate_x_expression(x_expr, curve, parsed)
    steps.append("compute_curve")

    # 4. Load data if available
    if data_path is None or not os.path.exists(data_path):
        return steps, None

    with open(data_path, "rb") as f:
        file_bytes = f.read()

    nmr_cfg  = parsed.get("nmr")
    spectra_cfg = parsed.get("spectra")

    if spectra_cfg is not None:
        spectra_data = load_spectra_data(file_bytes)
        steps.append("load_spectra_data")

        wl  = spectra_data["wavelengths"]
        fit_keys = list(logK_vals.keys())
        success, fitted, stats, msg = fit_spectra(
            parsed, network, spectra_data, params, logK_vals, fit_keys,
            x_expr, float(wl[0]), float(wl[-1]),
            tolerance=1e-4, maxiter=200, timeout_s=30.0,
        )
        steps.append(f"fit_spectra({'OK' if success else 'no-conv'})")

    elif nmr_cfg is not None:
        mode = nmr_cfg["mode"]
        exp_data = load_experimental_data(file_bytes)
        steps.append("load_nmr_data")

        fit_keys = list(logK_vals.keys())
        tol, maxiter = 1e-4, 500

        if mode == "shift":
            success, fitted, stats, msg = fit_nmr_shifts(
                parsed, network, exp_data, params, logK_vals,
                fit_keys, x_expr, tol, maxiter, timeout_s=30.0)
            steps.append(f"fit_nmr_shifts({'OK' if success else 'no-conv'})")

        elif mode == "integration":
            success, fitted, stats, msg = fit_nmr_integration(
                parsed, network, exp_data, params, logK_vals,
                fit_keys, x_expr, tol, maxiter, timeout_s=30.0)
            steps.append(f"fit_nmr_integration({'OK' if success else 'no-conv'})")

        elif mode == "mixed":
            success, fitted, stats, msg = fit_nmr_mixed(
                parsed, network, exp_data, params, logK_vals,
                fit_keys, x_expr, tol, maxiter, timeout_s=30.0)
            steps.append(f"fit_nmr_mixed({'OK' if success else 'no-conv'})")

    else:
        exp_data = load_experimental_data(file_bytes)
        steps.append("load_conc_data")

        fit_keys = list(logK_vals.keys())
        success, fitted, stats, msg = fit_parameters(
            parsed, network, exp_data, params, logK_vals,
            fit_keys, x_expr, tolerance=1e-4, maxiter=500)
        steps.append(f"fit_parameters({'OK' if success else 'no-conv'})")

    return steps, None


def run_kinetics(name, script_text, data_path, tdir):
    """Full kinetics pipeline: parse → ODE → fit (if data)."""
    steps = []

    parsed   = parse_script(script_text)
    steps.append("parse")

    logk_dict = build_kinetics_logk_dict(parsed)
    t_max     = float(parsed["plot_xmax"])
    kin_curve = compute_kinetics_curve(parsed, logk_dict, t_max, 200)
    steps.append("compute_kinetics_curve")

    if data_path is None or not os.path.exists(data_path):
        return steps, None

    with open(data_path, "rb") as f:
        file_bytes = f.read()

    nmr_cfg     = parsed.get("nmr")
    spectra_cfg = parsed.get("spectra")
    fit_keys    = list(logk_dict.keys())
    tol, maxiter = 1e-4, 500

    if spectra_cfg is not None:
        spectra_data = load_spectra_data(file_bytes)
        steps.append("load_spectra_data")
        wl = spectra_data["wavelengths"]
        success, fitted, stats, msg = fit_kinetics_spectra(
            parsed, logk_dict, spectra_data, fit_keys,
            t_max, 200, float(wl[0]), float(wl[-1]),
            tol, maxiter, timeout_s=30.0)
        steps.append(f"fit_kinetics_spectra({'OK' if success else 'no-conv'})")

    elif nmr_cfg is not None:
        mode     = nmr_cfg["mode"]
        nmr_data = load_experimental_data(file_bytes)
        steps.append("load_nmr_data")

        if mode == "shift":
            success, fitted, stats, msg = fit_kinetics_nmr_shifts(
                parsed, logk_dict, nmr_data, fit_keys,
                t_max, 200, tol, maxiter, timeout_s=30.0)
            steps.append(f"fit_kinetics_nmr_shifts({'OK' if success else 'no-conv'})")

        elif mode == "integration":
            success, fitted, stats, msg = fit_kinetics_nmr_integration(
                parsed, logk_dict, nmr_data, fit_keys,
                t_max, 200, tol, maxiter, timeout_s=30.0)
            steps.append(f"fit_kinetics_nmr_integration({'OK' if success else 'no-conv'})")

        elif mode == "mixed":
            success, fitted, stats, msg = fit_kinetics_nmr_mixed(
                parsed, logk_dict, nmr_data, fit_keys,
                t_max, 200, tol, maxiter, timeout_s=30.0)
            steps.append(f"fit_kinetics_nmr_mixed({'OK' if success else 'no-conv'})")

    else:
        exp_data = load_experimental_data(file_bytes)
        steps.append("load_conc_data")
        success, fitted, stats, msg = fit_kinetics(
            parsed, exp_data, logk_dict, fit_keys,
            t_max, 200, tol, maxiter, timeout_s=30.0)
        steps.append(f"fit_kinetics({'OK' if success else 'no-conv'})")

    return steps, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tutorials")

    scripts = sorted(
        f for f in os.listdir(tdir) if f.endswith(".txt") and f.startswith("tutorial_")
    )

    if not scripts:
        print(f"No tutorial_*.txt files found in {tdir}")
        sys.exit(1)

    print(f"\nEquinix tutorial tests — {len(scripts)} scripts in {tdir}\n")
    print(f"{'Tutorial':<20} {'Status':<8} {'Steps / Error'}")
    print("─" * 80)

    passed = failed = 0

    for fname in scripts:
        name = fname.replace(".txt", "")
        script_path = os.path.join(tdir, fname)
        input_stem  = INPUT_MAP.get(name)
        data_path   = os.path.join(tdir, input_stem + ".xlsx") if input_stem else None

        script_text = open(script_path).read()
        is_kinetics = any(
            (">" in line or "<>" in line) and ";" in line
            for line in script_text.splitlines()
            if not line.strip().startswith("#")
        )
        runner = run_kinetics if is_kinetics else run_equilibrium

        t0 = time.time()
        try:
            steps, _ = runner(name, script_text, data_path, tdir)
            elapsed  = time.time() - t0
            print(f"  {name:<20} {'PASS':<8} {' → '.join(steps)}  ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc().strip().split("\n")
            # Print last 3 lines of traceback for brevity
            err_lines = tb[-3:]
            print(f"  {name:<20} {'FAIL':<8} {e}")
            for line in err_lines:
                print(f"    {line}")
            failed += 1

    print("─" * 80)
    print(f"\n  {passed} passed, {failed} failed\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
