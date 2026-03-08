"""equinix_curve.py"""
import numpy as np
from scipy.optimize import least_squares
from equinix_network import solve_free_species, solve_equilibria_general
from equinix_network import compute_variable_curve, _sanitise_pct, _LN_LO, _LN_HI
from equinix_network import evaluate_variable_expression, resolve_variable_dependencies

__all__ = ['compute_curve', 'evaluate_x_expression', 'convert_exp_x', '_x_per_equiv', 'find_equiv_for_x', '_solid_frac_for_tkey', 'compute_single_point']


def compute_curve(parsed: dict, network: dict, logK_vals: dict, params: dict) -> dict:
    """
    Sweep equiv from 0 to maxEquiv, solving mass balances at each point.
    Handles both liquid titration (volume changes) and solid addition (volume fixed).
    """
    conc0_mM   = params["conc0"]
    V0         = params["V0_mL"]
    tit_names  = params["titrant_free_names"]
    tit_keys   = params["titrant_keys"]
    tit_mMs    = params["titrant_mMs"]      # {free_name: mM}  (0.0 for solid)
    tit_ratios = params["titrant_ratios"]   # {free_name: ratio} (only solid)
    is_solid   = params["titrant_is_solid"]
    tit_name   = tit_names[0]
    tit_mM     = tit_mMs[tit_name]          # 0.0 for solid
    maxEquiv   = params["maxEquiv"]
    nPts       = params["nPts"]
    primary    = params["primary_component"]

    n0        = {name: conc * V0 for name, conc in conc0_mM.items()}  # mmol
    xs        = np.linspace(0.0, float(maxEquiv), int(nPts))
    n_primary = n0.get(primary, 1.0)

    # Normalise solid ratios so they sum to 1
    if is_solid and tit_ratios:
        ratio_sum = sum(tit_ratios.values())
        solid_fractions = {tfree: tit_ratios[tfree] / ratio_sum for tfree in tit_names}
    else:
        solid_fractions = {}

    free_species = network["free_species"]
    all_species  = network["all_species"]

    out = {sp: np.zeros(len(xs)) for sp in all_species}
    out["equiv"]      = xs
    out["warn"]       = np.zeros(len(xs), dtype=bool)
    out["resid_norm"] = np.zeros(len(xs))

    all_comp_names   = list(n0.keys()) + tit_names
    totals_mM_arrays = {name: np.zeros(len(xs)) for name in all_comp_names}
    for tkey, tfree in zip(tit_keys, tit_names):
        totals_mM_arrays[tkey] = totals_mM_arrays[tfree]

    def safe_x0(totals):
        x0_vals = []
        for fs in free_species:
            total = totals.get(fs, 1e-20)
            free_fraction = 0.1 if total > 1e-6 else 0.5
            init_guess = max(total * free_fraction, 1e-15)
            x0_vals.append(np.log(init_guess))
        return np.clip(np.array(x0_vals), -30.0 + 1e-6, _LN_HI - 1e-6)

    x0 = None

    for i, eq in enumerate(xs):
        n_tit = eq * n_primary   # total mmol of titrant added at this point

        if is_solid:
            # ── Solid: volume is fixed ──────────────────────────────────
            V = V0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree in tit_names:
                frac     = solid_fractions.get(tfree, 1.0)
                n_species = n_tit * frac          # mmol of this solid species
                totals[tfree] = totals.get(tfree, 0.0) + (n_species / V) * 1e-3
        else:
            # ── Liquid: volume grows with titrant added ─────────────────
            V_add = n_tit / max(tit_mM, 1e-12)
            V     = V0 + V_add
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree, tkey in zip(tit_names, tit_keys):
                ratio    = tit_mMs[tfree] / max(tit_mM, 1e-12)
                tit_conc = (n_tit * ratio / V) * 1e-3
                totals[tfree] = totals.get(tfree, 0.0) + tit_conc

        # Store total concentrations in mM
        for name in list(n0.keys()) + tit_names:
            totals_mM_arrays[name][i] = totals.get(name, 0.0) * 1e3

        if x0 is None:
            x0 = safe_x0(totals)

        sol, free_concs = solve_free_species(totals, network, logK_vals, x0)

        residual_norm = float(np.linalg.norm(sol.fun))
        if residual_norm > 1e-6:
            x0_retry = np.clip(
                np.array([np.log(max(totals.get(fs, 1e-20) * 0.5, 1e-15)) for fs in free_species]),
                -30.0, 2.0
            )
            sol, free_concs = solve_free_species(totals, network, logK_vals, x0_retry)

        x0 = sol.x

        rn = float(np.linalg.norm(sol.fun))
        out["resid_norm"][i] = rn
        out["warn"][i]       = (not np.isfinite(rn)) or (rn > 1e-8)

        memo = {}
        for sp in all_species:
            if "_rigorous_concentrations" in network and sp in network["_rigorous_concentrations"]:
                out[sp][i] = network["_rigorous_concentrations"][sp] * 1e3
            else:
                out[sp][i] = network["species_conc_fn"](sp, free_concs, logK_vals, memo) * 1e3

    out["totals_mM"]    = totals_mM_arrays
    out["V0_mL"]        = V0
    out["mmol_titrant"] = xs * n_primary
    for tfree, tkey in zip(tit_names, tit_keys):
        if is_solid:
            frac = solid_fractions.get(tfree, 1.0)
        else:
            frac = tit_mMs[tfree] / max(tit_mM, 1e-12)
        out[f"mmol_{tkey}"] = xs * n_primary * frac
    return out


def evaluate_x_expression(expr: str, curve: dict, parsed: dict) -> tuple:
    """
    Evaluate the x-axis expression at every titration point.

    Namespace (all quantities dimensionally consistent as ratios or mM):

      Species names  (e.g. GHM, S0)
        → concentration in mM at each titration point (np.ndarray)

      $concentrations names  (e.g. G0, H0, Q0)
        → initial mmol = C0_mM * V0_mL  (scalar)
          So M0/G0 = mmol_titrant_added / mmol_G_initial = equivalents ✓

      $titrant name  (e.g. M0)
        → cumulative mmol of titrant added at each point (np.ndarray)

    Arithmetic operators + - * / ( ) are all supported.

    Returns (x_vals array, x_label string).
    Raises ValueError with a helpful message on failure.
    """
    all_species = set(k for k in curve.keys()
                      if k not in {"equiv", "warn", "resid_norm", "totals_mM", "V0_mL", "mmol_titrant"}
                      and not k.startswith("mmol_"))
    n           = len(curve["equiv"])
    V0_mL       = curve.get("V0_mL", 1.0)

    ns = {}

    # Species concentrations (mM arrays)
    for sp in all_species:
        ns[sp] = curve[sp]

    # $concentrations names → initial mmol (scalar)
    for cname, cval_mM in parsed["concentrations"].items():
        ns[cname] = cval_mM * V0_mL   # mmol

    # $titrant names → cumulative mmol added (array)
    for tkey in parsed["titrant"].keys():
        mmol_key = f"mmol_{tkey}"
        if mmol_key in curve:
            ns[tkey] = curve[mmol_key]
        else:
            ns[tkey] = curve.get("mmol_titrant", np.zeros(n))

    try:
        result = eval(expr, {"__builtins__": {}}, ns)
    except Exception as e:
        raise ValueError(f"Could not evaluate x expression '{expr}': {e}")

    result = np.broadcast_to(np.asarray(result, dtype=float), n).copy()
    return result, expr


# ─────────────────────────────────────────────
# SIDEBAR HELPERS
# ─────────────────────────────────────────────


def convert_exp_x(v_add_mL: np.ndarray, x_expr: str, parsed: dict,
                  params: dict, network: dict) -> np.ndarray:
    """
    Convert the experimental x column to the same x-axis expression used by the plot.

    Solid mode:  column A already contains x-axis values → return as-is.
    Liquid mode: column A is volume added (mL) → compute x from stock conc and expression.
    """
    is_solid = params.get("titrant_is_solid", False)

    if is_solid:
        # Column A values ARE the x-axis values — nothing to convert
        return np.asarray(v_add_mL, dtype=float).copy()

    # ── Liquid mode ──────────────────────────────────────────────────────
    tit_free_names = network["titrant_free_names"]
    tit_keys       = network["titrant_keys"]
    tit_mMs        = params["titrant_mMs"]
    primary        = params["primary_component"]
    conc0          = params["conc0"]
    V0             = params["V0_mL"]

    tit_mM_primary = tit_mMs[tit_free_names[0]]
    mmol_primary   = v_add_mL * tit_mM_primary

    ns = {}
    for root, cval_mM in conc0.items():
        key = root if root.endswith("0") else root + "0"
        ns[key] = cval_mM * V0

    for tfree, tkey in zip(tit_free_names, tit_keys):
        ratio    = tit_mMs[tfree] / max(tit_mM_primary, 1e-12)
        ns[tkey] = mmol_primary * ratio

    try:
        result = eval(x_expr, {"__builtins__": {}}, ns)
        result = np.broadcast_to(np.asarray(result, dtype=float), len(v_add_mL)).copy()
    except Exception:
        result = v_add_mL.copy()

    return result



def _x_per_equiv(x_expr, parsed, conc_vals, V0_mL,
                 titrant_free_names, titrant_keys, titrant_mMs, titrant_ratios,
                 is_solid, primary_component):
    """Return x(equiv=1) — the x-axis increment per equivalent."""
    V0       = V0_mL
    n_prim   = conc_vals.get(primary_component, 1.0) * V0   # mmol

    ns = {}
    # $concentrations → initial mmol (same as evaluate_x_expression)
    for cname, cval in parsed["concentrations"].items():
        ns[cname] = cval * V0

    if is_solid:
        ratio_sum = sum(titrant_ratios.values()) if titrant_ratios else 1.0
        for tfree, tkey in zip(titrant_free_names, titrant_keys):
            frac     = titrant_ratios.get(tfree, 1.0) / ratio_sum
            ns[tkey] = n_prim * frac        # mmol of this species at equiv=1
    else:
        tit_mM_prim = titrant_mMs[titrant_free_names[0]]
        for tfree, tkey in zip(titrant_free_names, titrant_keys):
            ratio    = titrant_mMs[tfree] / max(tit_mM_prim, 1e-12)
            ns[tkey] = n_prim * ratio       # mmol at equiv=1

    try:
        val = float(eval(x_expr, {"__builtins__": {}}, ns))
        return val if val > 0 else 1.0
    except Exception:
        return 1.0   # fallback: treat equiv ≡ x


def find_equiv_for_x(target_x, parsed, params):
    """
    Convert an x-axis value back to equivalents.
    Uses the same linear-coefficient approach as _x_per_equiv:
    equiv = target_x / x(equiv=1).
    """
    x_expr = parsed.get("plot_x_expr") or \
        f"{params['titrant_key']}/{list(parsed['concentrations'].keys())[0]}"
    xpe = _x_per_equiv(
        x_expr, parsed,
        params["conc0"], params["V0_mL"],
        params["titrant_free_names"], params["titrant_keys"],
        params["titrant_mMs"], params.get("titrant_ratios", {}),
        params.get("titrant_is_solid", False),
        params["primary_component"],
    )
    return target_x / xpe if xpe > 0 else target_x


def _solid_frac_for_tkey(tkey_num, params):
    """
    Return the normalised fraction of the solid titrant component whose
    script key is tkey_num (e.g. 'Mt').  Returns 1.0 if not found.
    """
    tit_keys   = params.get("titrant_keys", [])
    tit_names  = params.get("titrant_free_names", [])
    tit_ratios = params.get("titrant_ratios", {})
    ratio_sum  = sum(tit_ratios.values()) if tit_ratios else 1.0
    if ratio_sum == 0:
        return 1.0
    for tkey, tfree in zip(tit_keys, tit_names):
        if tkey == tkey_num:
            return tit_ratios.get(tfree, 1.0) / ratio_sum
    return 1.0


def compute_single_point(equiv, parsed, network, logK_vals, params, target_variable):
    """
    Compute theoretical concentration for a single titration point.
    Uses solve_equilibria_general directly — consistent with compute_curve.
    """
    try:
        conc0_mM  = params["conc0"]
        V0        = params["V0_mL"]
        tit_names = params["titrant_free_names"]
        tit_mMs   = params["titrant_mMs"]
        primary   = params["primary_component"]

        n0        = {name: conc * V0 for name, conc in conc0_mM.items()}
        n_primary = n0.get(primary, 1.0)
        equiv     = max(equiv, 0.0)

        n_tit = equiv * n_primary
        is_solid   = params.get("titrant_is_solid", False)
        tit_ratios = params.get("titrant_ratios", {})

        if is_solid:
            V = V0
            ratio_sum = sum(tit_ratios.values()) if tit_ratios else 1.0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree in tit_names:
                frac = tit_ratios.get(tfree, 1.0) / ratio_sum
                totals[tfree] = totals.get(tfree, 0.0) + (n_tit * frac / V) * 1e-3
        else:
            tit_mM = tit_mMs[tit_names[0]]
            V_add  = n_tit / max(tit_mM, 1e-12)
            V      = V0 + V_add
            if V <= 0:
                return 0.0
            totals = {name: (n0[name] / V) * 1e-3 for name in n0}
            for tfree, tkey in zip(tit_names, params["titrant_keys"]):
                ratio = tit_mMs[tfree] / max(tit_mM, 1e-12)
                totals[tfree] = totals.get(tfree, 0.0) + (n_tit * ratio / V) * 1e-3

        # Solve with the general solver
        equilibria  = network.get("equilibria", [])
        all_species = network["all_species"]
        concs_M, _, _ = solve_equilibria_general(totals, equilibria, all_species, logK_vals)

        # Convert to mM and read target
        concs_mM = {sp: concs_M[sp] * 1e3 for sp in all_species}

        # Direct species
        if target_variable in concs_mM:
            val = concs_mM[target_variable]
            return float(val) if np.isfinite(val) else 0.0

        # Expression variable ($variables section)
        variables = parsed.get("variables", {})
        if target_variable in variables:
            var_order = resolve_variable_dependencies(variables)
            variable_values = {}
            for var_name in var_order:
                variable_values[var_name] = evaluate_variable_expression(
                    variables[var_name], concs_mM, variable_values
                )
            val = variable_values.get(target_variable, 0.0)
            return float(val) if np.isfinite(val) else 0.0

        return 0.0

    except Exception:
        return 0.0


# ─────────────────────────────────────────────