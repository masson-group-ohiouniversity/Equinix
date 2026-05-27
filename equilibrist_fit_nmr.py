# -*- coding: utf-8 -*-
"""equilibrist_fit_nmr.py"""
import re
import time
import numpy as np
from scipy.optimize import minimize
from equilibrist_network import compute_variable_curve, _sanitise_pct
from equilibrist_curve import convert_exp_x, find_equiv_for_x, compute_single_point
from equilibrist_curve import compute_curve, evaluate_x_expression
from equilibrist_parser import constraints_penalty
from equilibrist_shift_constraints import solve_shifts, get_per_column_bounds

__all__ = ['compute_nmr_curves', '_get_species_for_target', '_hessian_errors', 'fit_nmr_shifts', '_resolve_c', '_nmr_integration_backCalc', 'fit_nmr_integration', 'fit_nmr_mixed']


def compute_nmr_curves(nmr_cfg: dict, parsed: dict, curve: dict,
                        network: dict, x_vals: np.ndarray,
                        delta_bound: dict) -> dict:
    """
    Compute theoretical NMR chemical shift curves for all targets.

    Fast-exchange (shift mode):
        δ_obs(x) = Σ_i  [species_i](x) / target_total(x)  × δ_i

    Parameters
    ----------
    nmr_cfg    : parsed["nmr"]  {"mode": "shift", "targets": ["Gtot"]}
    curve      : concentration dict from compute_curve  (mM arrays)
    delta_bound: {target: {species: δ_value}}   — pure-species shifts
                 e.g. {"Gtot": {"G": 8.0, "GH": 12.5}}

    Returns
    -------
    {"Gtot": np.ndarray of δ_obs values}   (one entry per target)
    """
    variables  = parsed.get("variables", {})
    all_species = network["all_species"]
    out = {}

    for target in nmr_cfg["targets"]:
        shifts = delta_bound.get(target, {})
        if not shifts:
            continue

        # Numerator: sum of [sp](x) * δ_sp
        numerator = np.zeros_like(x_vals)
        # Denominator: total concentration of target species
        denominator = np.zeros_like(x_vals)

        for sp, delta in shifts.items():
            if sp in curve:
                c_sp = curve[sp]
            elif sp in variables:
                # Evaluate variable expression
                c_sp = compute_variable_curve(sp, variables, curve, network, x_vals)
            else:
                c_sp = np.zeros_like(x_vals)
            numerator   += c_sp * delta
            denominator += c_sp

        # Avoid division by zero
        safe_denom = np.where(denominator > 1e-20, denominator, 1e-20)
        out[target] = numerator / safe_denom

    return out


def _get_species_for_target(target: str, parsed: dict, network: dict) -> list:
    """
    Return [(coeff, species), ...] for the target variable (or bare species).

    Parses stoichiometric coefficients so that weighted NMR averages are exact:
      Gtot = G + GH + GH2     -> [(1,'G'), (1,'GH'), (1,'GH2')]
      Htot = H + GH + 2*GH2  -> [(1,'H'), (1,'GH'), (2,'GH2')]
      bare species G           -> [(1,'G')]
    """
    variables   = parsed.get("variables", {})
    all_species = network["all_species"]

    if target in variables:
        expr  = variables[target]
        parts = [p.strip() for p in expr.split("+") if p.strip()]
        result = []
        for part in parts:
            m = re.match(r'^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$', part)
            if m:
                coeff, sp = float(m.group(1)), m.group(2)
            else:
                coeff, sp = 1.0, part
            if sp in all_species:
                result.append((coeff, sp))
        return result
    elif target in all_species:
        return [(1.0, target)]
    return []


def _hessian_errors(objective_fn, x_best: np.ndarray, ssr_best: float,
                    n_data: int, n_params: int, step: float = 1e-3):
    """
    Estimate ±log K errors from the finite-difference Hessian of the objective.

    Returns a tuple ``(errors_by_index, cov_matrix)``:
      * ``errors_by_index`` — ``{i: σᵢ}`` mapping the position in
        ``x_best`` to the marginal standard error.  Same as the legacy
        return value used by callers.
      * ``cov_matrix`` — the full ``n × n`` covariance matrix as a
        numpy array (or ``None`` on numerical failure).  Off-diagonal
        elements give the parameter covariances that the diagnostics
        layer normalises to a correlation matrix for the heatmap.

    Method
    ──────
    For a least-squares objective  f(θ) = SSR(θ),  the Hessian at the minimum is
    approximately  H ≈ 2 J'J  (Gauss-Newton), and the covariance matrix is:

        Cov(θ) = σ² × (H/2)⁻¹  =  [SSR/(n-p)] × (J'J)⁻¹

    When the Jacobian is unavailable (Nelder-Mead), we estimate H by central
    finite differences of the scalar objective:

        H[i,i]  ≈  [f(θ+hᵢ) + f(θ-hᵢ) - 2f(θ)] / hᵢ²   (diagonal)
        H[i,j]  ≈  [f(θ+hᵢ+hⱼ) - f(θ+hᵢ-hⱼ)
                    - f(θ-hᵢ+hⱼ) + f(θ-hᵢ-hⱼ)] / (4hᵢhⱼ)  (off-diagonal)

    σ² = SSR / (n - p)  is the reduced chi-squared.
    Error on parameter i = sqrt(Cov[i,i]) in log K units.
    """
    n = len(x_best)
    if n == 0 or n_data <= n_params:
        return {}, None

    sigma2 = max(ssr_best / (n_data - n_params), 1e-30)
    h = np.abs(x_best) * step
    h[h < 1e-6] = step   # minimum step

    try:
        f0 = float(objective_fn(x_best))   # evaluate at minimum (may differ from ssr_best)
        H  = np.zeros((n, n))

        # Diagonal elements
        for i in range(n):
            ei      = np.zeros(n); ei[i] = h[i]
            fp      = objective_fn(x_best + ei)
            fm      = objective_fn(x_best - ei)
            H[i, i] = (fp + fm - 2 * f0) / (h[i] ** 2)

        # Off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = h[i]
                ej = np.zeros(n); ej[j] = h[j]
                fpp = objective_fn(x_best + ei + ej)
                fpm = objective_fn(x_best + ei - ej)
                fmp = objective_fn(x_best - ei + ej)
                fmm = objective_fn(x_best - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])

        # Regularize: ensure H is positive definite
        eigvals = np.linalg.eigvalsh(H)
        if np.any(eigvals <= 0):
            H += np.eye(n) * max(-eigvals.min() * 1.01, 1e-10)

        cov = np.linalg.inv(H / 2.0) * sigma2
        errors = {}
        for i in range(n):
            errors[i] = float(np.sqrt(max(cov[i, i], 0.0)))
        return errors, cov
    except Exception:
        return {}, None


def fit_nmr_shifts(parsed: dict, network: dict, nmr_data: dict,
                   params: dict, logK_vals: dict, fit_keys: list,
                   x_expr: str, tolerance: float, maxiter: int,
                   timeout_s: float = 30.0, constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None,
                   *, compute_hessian: bool = True):
    """
    Fit equilibrium constants + per-signal pure-species shifts to NMR data.

    General fast-exchange model with stoichiometric weights:
        δ_obs = Σᵢ (nᵢ·[Sᵢ]) / Σⱼ (nⱼ·[Sⱼ])  × δᵢ
              = δ_free + Σᵢ₌₁..N-1  Fᵢ(x) × Δδᵢ

    where nᵢ are stoichiometric coefficients from the $variables expression
    (e.g. Htot = H + GH + 2*GH2 gives n_GH2=2) and Fᵢ = nᵢ·[Sᵢ] / Σnⱼ·[Sⱼ].

    δ_free is read from the first data point of each signal.  For signals where
    V=0 is missing (e.g. host not present at start), the first available point
    is used as reference and the theoretical curve is anchored there too.

    The Δδ vector is solved analytically by linear least squares for each trial
    of K — only K values are optimised by Nelder-Mead.
    """
    from scipy.optimize import minimize
    nmr_cfg = parsed["nmr"]

    # ── Solid mode: retrieve column-A header ─────────────────────────────────
    _x_col_header = nmr_data.get("_x_col_header", "")

    # ── Map columns to target variables ──────────────────────────────────────
    col_to_target = {}
    for col in nmr_data:
        if col.startswith("_"): continue
        for tgt in nmr_cfg["targets"]:
            if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                col_to_target[col] = tgt; break
        else:
            col_to_target[col] = nmr_cfg["targets"][0] if nmr_cfg["targets"] else col

    # ── δ_free: first available data point per signal ─────────────────────────
    # (handles host signals where V=0 row is NaN/missing)
    delta_free = {}
    x_free_val = {}   # x-axis value at which δ_free was measured
    for col, col_data in nmr_data.items():
        if col.startswith("_"): continue
        delta_free[col] = float(col_data["y"][0])
        # x value at first point (used to anchor theoretical curve)
        x_arr = convert_exp_x(col_data["v_add_mL"], x_expr, parsed, params, network,
                               x_col_header=_x_col_header)
        x_free_val[col] = float(x_arr[0])

    first_col = next((c for c in nmr_data if not c.startswith("_")), None)
    if first_col is None:
        return False, {}, {}, "No NMR data columns found"


    # ── Concentration fitting setup ──────────────────────────────────────────
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)
    _CONC_MIN = 0.0

    _root_to_cname_nmr = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_nmr[_r] = _cn

    def _cb_nmr(root, default_val, sv_override=None):
        cname = _root_to_cname_nmr.get(root, root)
        sv  = sv_override if sv_override is not None else \
              parsed.get("concentrations", {}).get(cname, default_val)
        lo, hi = parsed.get("conc_bounds", {}).get(cname,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    def _tb_nmr(tkey, default_val, sv_override=None):
        sv  = sv_override if sv_override is not None else \
              parsed.get("titrant", {}).get(tkey, default_val)
        lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    # Seed from script values so optimizer starts inside bounds regardless of sidebar
    _root_to_cname_x0 = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_x0[_r] = _cn
    def _script_val(root):
        cname = _root_to_cname_x0.get(root, root)
        sv = parsed.get("concentrations", {}).get(cname)
        return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
    _x0_conc_nmr    = [_script_val(cn) for cn in fit_conc_keys]
    _x0_titrant_nmr = []
    for _tk in fit_titrant_keys:
        _tfree = _tk[:-1] if (_tk.endswith("t") or _tk.endswith("0")) else _tk
        _x0_titrant_nmr.append(params["titrant_mMs"].get(_tfree, 10.0))
    _bounds_conc_nmr    = [_cb_nmr(cn, _x0_conc_nmr[i])    for i, cn in enumerate(fit_conc_keys)]
    _bounds_titrant_nmr = [_tb_nmr(tk, _x0_titrant_nmr[i]) for i, tk in enumerate(fit_titrant_keys)]

    # Per-K soft bounds from the script's 'from X to Y' syntax (defaults if absent).
    # Honored as a quadratic penalty in objective() since Nelder-Mead ignores `bounds`.
    _DEFAULT_LO_K, _DEFAULT_HI_K = -2.0, 14.0
    _eq_by_kname_nmr = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
    def _resolve_k_bound_nmr(k, attr, default):
        v = _eq_by_kname_nmr.get(k, {}).get(attr)
        return float(v) if v is not None else float(default)
    _k_lo_nmr = np.array([_resolve_k_bound_nmr(k, "logK_lo", _DEFAULT_LO_K) for k in fit_keys])
    _k_hi_nmr = np.array([_resolve_k_bound_nmr(k, "logK_hi", _DEFAULT_HI_K) for k in fit_keys])

    def _update_bds_nmr(fitted_x):
        """Recentre bounds to ±20% of current fitted values after each pass."""
        for _i, _cn in enumerate(fit_conc_keys):
            _sv = float(np.clip(fitted_x[_n_k + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_conc_nmr[_i] = _cb_nmr(_cn, _sv, sv_override=_sv)
        for _i, _tk in enumerate(fit_titrant_keys):
            _sv = float(np.clip(fitted_x[_n_k + _n_c + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_titrant_nmr[_i] = _tb_nmr(_tk, _sv, sv_override=_sv)

    def _make_simplex_steps_nmr():
        """Step = full feasible range (hi - lo) per concentration/titrant dim."""
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else 1.5))
        c_steps = np.array([_bounds_conc_nmr[i][1]    - _bounds_conc_nmr[i][0]
                            for i in range(_n_c)])
        t_steps = np.array([_bounds_titrant_nmr[i][1] - _bounds_titrant_nmr[i][0]
                            for i in range(len(fit_titrant_keys))])
        return np.concatenate([k_steps, c_steps, t_steps]) \
               if len(c_steps) or len(t_steps) else k_steps

    def _make_params_nmr(trial):
        if not fitting_concs:
            return params
        n_k = len(fit_keys)
        p = dict(params)
        p["conc0"] = dict(params["conc0"])
        p["titrant_mMs"] = dict(params.get("titrant_mMs", {}))
        for i, cn in enumerate(fit_conc_keys):
            p["conc0"][cn] = float(np.clip(trial[n_k + i],
                                           _bounds_conc_nmr[i][0],
                                           _bounds_conc_nmr[i][1]))
        for i, tk in enumerate(fit_titrant_keys):
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            p["titrant_mMs"][tfree] = float(np.clip(trial[n_k + len(fit_conc_keys) + i],
                                                     _bounds_titrant_nmr[i][0],
                                                     _bounds_titrant_nmr[i][1]))
        return p

    def _simulate(logk_trial, _params=None):
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_curve(parsed, network, lk, _params or params)
        except Exception:
            return None

    def _x_for_col(col_data, _params=None):
        return convert_exp_x(col_data["v_add_mL"], x_expr, parsed, _params or params, network,
                             x_col_header=_x_col_header)

    def _build_fraction_matrix(c, x_exp, sp_coeffs, x_free=None,
                                df_theoretical=None, include_ref=False):
        """
        Build design matrix X of shape (n_pts, N) or (n_pts, N-1).

        Fᵢ(x) = nᵢ·[Sᵢ](x) / Σⱼ nⱼ·[Sⱼ](x)   = mole fraction of species i

        If ``include_ref=False`` (default / legacy): builds X with N−1 columns
        for the non-reference species (sp_coeffs[1:]), used in the dd_ref ≡ 0
        formulation.

        If ``include_ref=True``: builds X with N columns for ALL species
        in the target (including sp_coeffs[0]), used in the noref formulation
        where every species has its own free dd parameter.

        If ``x_free`` is given, subtract the theoretical δ at that reference
        point so the regression is for Δδ relative to the same anchor as the
        data.
        """
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        n_pts = len(x_exp)

        # Weighted denominator: Σ nⱼ·[Sⱼ]
        denom = np.zeros(n_pts)
        for coeff, sp in sp_coeffs:
            denom += coeff * np.interp(x_exp, x_sim, c.get(sp, np.zeros_like(x_sim)))
        denom = np.maximum(denom, 1e-20)

        # Choose which species get columns
        if include_ref:
            cols_sp = list(sp_coeffs)             # all species
        else:
            cols_sp = list(sp_coeffs[1:])         # non-reference only

        # X[:,i] = nᵢ·[Sᵢ] / denom
        X = np.zeros((n_pts, len(cols_sp)))
        for i, (coeff, sp) in enumerate(cols_sp):
            X[:, i] = coeff * np.interp(x_exp, x_sim, c.get(sp, np.zeros_like(x_sim))) / denom

        # If a reference x-point is given, subtract the fraction at that point
        # so the model predicts Δδ = 0 at the first measurement (as data does)
        if x_free is not None:
            X_ref = np.zeros(len(cols_sp))
            denom_ref = sum(coeff * float(np.interp(x_free, x_sim, c.get(sp, np.zeros_like(x_sim))))
                            for coeff, sp in sp_coeffs)
            denom_ref = max(denom_ref, 1e-20)
            for i, (coeff, sp) in enumerate(cols_sp):
                X_ref[i] = coeff * float(np.interp(x_free, x_sim,
                                                     c.get(sp, np.zeros_like(x_sim)))) / denom_ref
            X = X - X_ref[np.newaxis, :]

        return X

    # Capture user shift constraints once for this fit (cheap to look up)
    _shift_constraints = parsed.get("shift_constraints", []) or []

    # Per-column bounds (the 'shift: -0.8, free, 1.0, ...' syntax).  Build a
    # {col_name: bound_value_or_None} map for use by the inner solver.
    _per_col_bounds = get_per_column_bounds(_shift_constraints)
    _col_to_bound = {}
    if _per_col_bounds is not None:
        _shift_cols_in_order = [c for c in nmr_data if not c.startswith("_")]
        if len(_per_col_bounds) != len(_shift_cols_in_order):
            return False, {}, {}, (
                f"Per-column shift bounds list has {len(_per_col_bounds)} entries "
                f"but the data has {len(_shift_cols_in_order)} shift columns. "
                f"Use 'free' for columns you don't want to constrain."
            )
        _col_to_bound = dict(zip(_shift_cols_in_order, _per_col_bounds))

    # Pinned intrinsic shifts: $nmr 'read:' species + sheet-2 known shifts.
    # Convert absolute ppm in sheet 2 to dd values via dd_X[col] = δ_X − δ_obs(x_free, col).
    # Built as {col_name: {species: dd_value}} for per-column lookup.
    _read_species  = list(nmr_cfg.get("read", []) or [])
    _known_shifts  = nmr_data.get("_known_shifts", {}) or {}
    _col_to_pinned = {}
    if _read_species:
        _shift_cols_in_order = [c for c in nmr_data if not c.startswith("_")]
        for sp in _read_species:
            if sp not in _known_shifts:
                return False, {}, {}, (
                    f"$nmr 'read: {sp}' but species '{sp}' not found in sheet 2 of "
                    f"the data file (expected row with '{sp}' in column A and "
                    f"{len(_shift_cols_in_order)} chemical-shift values to its right)."
                )
            vals = _known_shifts[sp]
            if len(vals) != len(_shift_cols_in_order):
                return False, {}, {}, (
                    f"$nmr 'read: {sp}': sheet 2 has {len(vals)} values for '{sp}' "
                    f"but the data has {len(_shift_cols_in_order)} shift columns. "
                    f"Provide one value per shift column, in the same order."
                )
            for k, col in enumerate(_shift_cols_in_order):
                dd_val = float(vals[k]) - float(delta_free[col])
                _col_to_pinned.setdefault(col, {})[sp] = dd_val

    # NMR-level noref flag — when True, every solve_shifts call is run in
    # noref mode (no auto-pin to reference), requiring at least one read:
    # anchor.  Default False preserves legacy "Δδ relative to ref" behavior.
    _nmr_noref = bool(nmr_cfg.get("noref", False))

    def _analytic_delta(X, delta_obs_rel, sp_coeffs=None, column_bound=None,
                        pinned_dd=None):
        """Solve the per-column shift sub-problem.  X must have a column for
        every species in sp_coeffs (build with include_ref=True).
        Returns (dd, calc, ssr) where dd[i] corresponds to sp_coeffs[i]."""
        if X.shape[1] == 0:
            return np.array([]), np.zeros_like(delta_obs_rel), float(np.sum(delta_obs_rel**2))
        if sp_coeffs is None:
            # Legacy fall-through (no species info): bare lstsq
            dd = np.linalg.lstsq(X, delta_obs_rel, rcond=None)[0]
            calc = X @ dd
            return dd, calc, float(np.sum((delta_obs_rel - calc)**2))
        species = [sp for _, sp in sp_coeffs]
        ref_sp  = sp_coeffs[0][1]
        try:
            return solve_shifts(X, delta_obs_rel, species, ref_sp,
                                _shift_constraints,
                                column_bound=column_bound,
                                pinned_dd=pinned_dd,
                                noref=_nmr_noref)
        except ValueError:
            # Bad constraint topology — fall back to unconstrained lstsq so
            # the fit still runs (the parser warning will alert the user).
            dd = np.linalg.lstsq(X, delta_obs_rel, rcond=None)[0]
            calc = X @ dd
            return dd, calc, float(np.sum((delta_obs_rel - calc)**2))

    def objective(trial):
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        # Penalty for log-K outside per-K bounds
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None: return 1e12
        total_ssr = 0.0
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_exp         = _x_for_col(col_data, _p)
            _xfv          = float(x_exp[0])
            delta_obs_rel = col_data["y"] - delta_free[col]
            X = _build_fraction_matrix(c, x_exp, sp_coeffs, x_free=_xfv, include_ref=True)
            _, _, ssr = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            total_ssr += ssr
        cp = constraints_penalty(constraints or [], lk, ssr_scale=total_ssr)
        return total_ssr + cp + conc_penalty

    def data_objective(full_trial):
        """Data-only SSR for Hessian — accepts full parameter vector (logK + conc)."""
        _p = _make_params_nmr(full_trial) if fitting_concs else params
        c = _simulate(full_trial[:len(fit_keys)], _p)
        if c is None: return 1e12
        total_ssr = 0.0
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_exp         = _x_for_col(col_data, _p)
            _xfv          = float(x_exp[0])
            delta_obs_rel = col_data["y"] - delta_free[col]
            X = _build_fraction_matrix(c, x_exp, sp_coeffs, x_free=_xfv, include_ref=True)
            _, _, ssr = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))
            total_ssr += ssr
        return total_ssr

    import time
    x0  = np.concatenate([
        [logK_vals[k] for k in fit_keys],
        _x0_conc_nmr,
        _x0_titrant_nmr,
    ])
    n_p    = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _n_k   = len(fit_keys)
    _n_c   = len(fit_conc_keys)

    # ── Global timer starts here — covers Phase 0 AND the re-fit loop ───
    _global_start  = time.time()

    # ── Phase 0: warm-start logK before loop ────────────────────────────
    if fitting_concs and _n_k > 0:
        _sp0 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                          for i in range(_n_k)])
        class _TimeoutP0(Exception): pass
        def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s):
            if time.time() - _t0 > _tlim * 0.4:  # use at most 40% of budget on warm-start
                raise _TimeoutP0()
            return objective(np.concatenate([lk, x0[_n_k:]]))
        try:
            _r0 = minimize(_ph0_obj, x0[:_n_k], method="Nelder-Mead",
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp0})
            x0 = np.concatenate([_r0.x, x0[_n_k:]])
        except (_TimeoutP0, Exception):
            pass  # use whatever x0 we have

    # ── Re-fit loop ──────────────────────────────────────────────────────
    n_passes       = 200 if fitting_concs else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True; break

        # After the warm-start pass, K is near its optimum. Use a tight K step
        # (0.1 log-units) so subsequent passes refine concentrations without
        # wastefully re-exploring K space from scratch every time.
        _base_steps = _make_simplex_steps_nmr()
        if _pass > 0 and fitting_concs:
            _base_steps[:_n_k] = 0.1
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])
        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0}

        def _obj_timed(trial, _tracker=_bt, _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            f = objective(trial)
            if f < _tracker["f"]: _tracker["f"] = f; _tracker["x"] = trial.copy()
            if time.time() - _t0 > _tlim: raise _Timeout()
            return f

        try:
            result = minimize(_obj_timed, x0, method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult(); _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])

        if fitting_concs:
            x0_prev = x0.copy()
            x0 = result.x.copy(); best_x_global = x0.copy()
            _update_bds_nmr(x0)
            if _any_timed_out: break
            if np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:])) < tolerance: break
        else:
            best_x_global = result.x.copy()

    class _BestResult:
        x = best_x_global; success = getattr(result, "success", False)
        fun = getattr(result, "fun", np.inf); nit = _total_nit
    result = _BestResult()

    fitted_logKs = {fit_keys[i]: result.x[i] for i in range(len(fit_keys))}
    _fitted_concs_nmr    = {fit_conc_keys[i]: float(result.x[len(fit_keys) + i])
                            for i in range(len(fit_conc_keys))}
    _fitted_titrants_nmr = {fit_titrant_keys[i]: float(result.x[len(fit_keys) + len(fit_conc_keys) + i])
                             for i in range(len(fit_titrant_keys))}
    _params_final = _make_params_nmr(result.x) if fitting_concs else params

    # ── Final pass: Δδ vectors and statistics ────────────────────────────────
    c_final = _simulate(result.x[:len(fit_keys)], _params_final)
    pure_shifts     = {}
    delta_vecs_all  = {}   # {col: {sp: Δδ_relative_to_free}}
    delta_bound_all = {}   # 1:1 compat
    ref_corrections = {}   # {col: float} — correction at reference x-point
    all_residuals   = []
    all_y_obs       = []
    # v2: per-column observed and calculated arrays for bootstrap resampling
    per_col_y_obs   = {}   # {col: ndarray of Δδ_obs_rel}
    per_col_y_calc  = {}   # {col: ndarray of Δδ_calc_rel}
    per_col_x       = {}   # {col: ndarray of x-axis values}

    if c_final is not None:
        for col, col_data in nmr_data.items():
            if col.startswith("_"): continue
            tgt       = col_to_target.get(col, nmr_cfg["targets"][0])
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue

            x_exp         = _x_for_col(col_data, _params_final)
            delta_obs     = col_data["y"]
            df_exp        = delta_free[col]
            delta_obs_rel = delta_obs - df_exp

            X  = _build_fraction_matrix(c_final, x_exp, sp_coeffs, x_free=x_free_val[col], include_ref=True)
            dd, delta_calc_rel, _ = _analytic_delta(X, delta_obs_rel, sp_coeffs, _col_to_bound.get(col), _col_to_pinned.get(col))

            all_residuals.extend((delta_obs_rel - delta_calc_rel).tolist())
            all_y_obs.extend(delta_obs_rel.tolist())   # use Δδ, not absolute ppm
            per_col_y_obs[col]  = np.asarray(delta_obs_rel,  dtype=float)
            per_col_y_calc[col] = np.asarray(delta_calc_rel, dtype=float)
            per_col_x[col]      = np.asarray(x_exp,          dtype=float)

            # In the new convention: dd[i] = δ_i_absolute − δ_obs(V=0).
            # `dd` is the full per-species vector (same order as sp_coeffs).
            # In legacy mode (auto-pin ref to dd=0) dd[ref]=0; in noref mode
            # dd[ref] is fitted from the data.
            sp_dd = {sp: float(dd[i]) if i < len(dd) else 0.0
                     for i, (_, sp) in enumerate(sp_coeffs)}
            # When the absolute shift scale is NOT anchored by a read: pin,
            # subtract dd[ref] from every species so the math reference reads
            # as 0.  This makes legacy and noref-no-read produce identical
            # "Δδ relative to ref" output.  (When anchored, leave dd alone so
            # we can recover absolute δ as δ_obs(V=0) + dd.)
            if not (_nmr_noref and _read_species):
                ref_sp = sp_coeffs[0][1]
                ref_dd = sp_dd.get(ref_sp, 0.0)
                if abs(ref_dd) > 1e-12:
                    sp_dd = {k: v - ref_dd for k, v in sp_dd.items()}
            delta_vecs_all[col] = sp_dd

            if len(sp_coeffs) == 2:
                # 2-species target: store the non-ref dd in delta_bound_all
                non_ref_sp = sp_coeffs[1][1]
                delta_bound_all[col] = sp_dd.get(non_ref_sp, 0.0)

            # Pure-species shifts:
            # Always stored as dd values (= δ_absolute − δ_obs(V=0)).
            # Display/export adds δ_obs(V=0) back to recover absolute when
            # the shift scale is anchored (noref + at least one read:).
            # ref_corrections[col] is consumed by app.py's back-calc-to-
            # concentrations machinery. It is defined as
            #     ref_correction = Σᵢ Fᵢ(x_ref) × ddᵢ   (over ALL species,
            # including the free / reference species).  In legacy & noref-
            # without-read modes the normalization above pins dd[ref]=0, so
            # the ref-species term contributes zero and the sum can be
            # restricted to non-ref species.  In noref+read mode dd[ref]
            # may be nonzero — we must include it in the sum.
            x_sim_fp, _ = evaluate_x_expression(x_expr, c_final, parsed)
            denom_at_ref = max(sum(
                coeff * float(np.interp(x_free_val[col], x_sim_fp,
                              c_final.get(sp, np.zeros_like(x_sim_fp))))
                for coeff, sp in sp_coeffs), 1e-20)
            ref_corrections[col] = sum(
                coeff * float(np.interp(x_free_val[col], x_sim_fp,
                              c_final.get(sp, np.zeros_like(x_sim_fp)))) / denom_at_ref
                * sp_dd.get(sp, 0.0)
                for coeff, sp in sp_coeffs   # all species, incl. ref
            )
            if tgt not in pure_shifts:
                pure_shifts[tgt] = {}
            pure_shifts[tgt][col] = dict(sp_dd)

    residuals = np.array(all_residuals)
    y_obs     = np.array(all_y_obs)
    ssr  = float(np.sum(residuals**2))
    sst  = float(np.sum((y_obs - y_obs.mean())**2)) if len(y_obs) > 1 else 1.0
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

    # Parameter errors via finite-difference Hessian (data-only objective, never penalised).
    # Skipped when ``compute_hessian=False`` (bootstrap workers).  Defaults
    # to empty dict / None placeholders so downstream code that reads
    # param_errors / param_cov doesn't have to special-case the absence.
    _n_total_params_shift = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    param_errors_shift = {}
    _cov_mat = None
    _cov_names_shift = list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(data_objective, result.x, ssr, len(residuals), _n_total_params_shift)
        param_errors_shift = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors_shift[fit_conc_keys[_i]] = _err_idx[_idx]
        for _i in range(len(fit_titrant_keys)):
            _idx = len(fit_keys) + len(fit_conc_keys) + _i
            if _idx in _err_idx: param_errors_shift[fit_titrant_keys[_i]] = _err_idx[_idx]

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "ssr":             ssr,
        "n_points":        len(residuals),
        "n_params":        len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys),
        "param_values":    fitted_logKs,
        "param_errors":    param_errors_shift,
        # Full covariance matrix for the correlation-heatmap and t-test
        # diagnostics.  ``param_cov_names`` is the row/column order.
        "param_cov":       _cov_mat,
        "param_cov_names": _cov_names_shift,
        "pure_shifts":     pure_shifts,        # dd values (= δ − δ_obs(V=0))
        "pure_shifts_anchored": bool(_nmr_noref and _read_species),
        "nmr_noref":             bool(_nmr_noref),
        "delta_vecs_all":  delta_vecs_all,
        "delta_bound_all": delta_bound_all,
        "delta_free":      delta_free,
        "x_free_val":      x_free_val,
        "col_to_target":   col_to_target,
        "ref_corrections": ref_corrections,
        "fit_mode":        "shift",
        "fitted_concs":    _fitted_concs_nmr,
        "fitted_titrants": _fitted_titrants_nmr,
        "n_iter":          getattr(result, "nit", 0),
        "timed_out":       not getattr(result, "success", True),
        "sp_concs":        {},      # not used in shift mode; keeps display logic uniform
        "col_to_sp":       {},
        "col_to_nH":       {},
        # v2 diagnostics — flat residual arrays
        "y_obs":           np.asarray(y_obs,     dtype=float),
        "y_calc":          np.asarray(y_obs - residuals, dtype=float),
        "residuals":       np.asarray(residuals, dtype=float),
        # v2 bootstrap — per-column arrays for residual resampling
        "per_col_y_obs":   per_col_y_obs,
        "per_col_y_calc":  per_col_y_calc,
        "per_col_x":       per_col_x,
    }
    _to   = stats.get("timed_out", False)
    _r2   = stats.get("r_squared", 0.0)
    _conv = result.success or ssr < 1e-6 or (_to and _r2 >= 0.99)
    if _to and _r2 >= 0.99:
        stats["timed_out"] = False   # good enough — suppress warning
    _msg = "NMR shift fit complete"
    return _conv, fitted_logKs, stats, _msg



def _resolve_c(c: dict, sp: str, parsed: dict, x_grid: np.ndarray) -> np.ndarray:
    """
    Return the concentration array for `sp` on x_grid.
    Works for both real species (direct lookup) and $variables (sum of members).
    """
    if sp in c:
        return c[sp]
    variables = parsed.get("variables", {})
    if sp in variables:
        expr    = variables[sp]
        total   = np.zeros(len(x_grid))
        for part in expr.split("+"):
            part = part.strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$", part)
            coeff, member = (float(m.group(1)), m.group(2)) if m else (1.0, part)
            total += coeff * c.get(member, np.zeros(len(x_grid)))
        return total
    return np.zeros(len(x_grid))


def _nmr_integration_backCalc(nmr_data: dict, n_H_list: list, params: dict,
                               network: dict, x_expr: str, parsed: dict,
                               x_col_header: str = "") -> dict:
    """
    Convert per-spectrum-normalised NMR integrations to absolute concentrations.

    Physical model
    ──────────────
    The NMR software normalises each spectrum so that the primary component
    signal (e.g. G) integrates to a fixed reference value (typically 1.0 × n_H).
    All other integrals in that spectrum are therefore relative to [G] at that
    titration point.

    For any species Sp:
        I_sp / n_H_sp  =  [Sp](x) / [G](x)     (ratio in the normalised spectrum)

    G mass-balance:
        [G_total](x)  =  Σ_{sp containing G}  stoich(G, sp) × [Sp](x)
                      =  [G](x) × Σ_{sp containing G}  stoich(G, sp) × ratio_sp(x)

    where  ratio_sp(x) = mean over duplicate signals of I_sp/n_H_sp.

    Therefore:
        [G](x)   = [G_total](x) / denom(x)
        [Sp](x)  = ratio_sp(x)  × [G](x)
        [G_total](x) = G0 × V0/(V0+V_add)   (liquid)  or  G0  (solid)

    Returns
    ───────
    {sp_name: (x_arr, c_bc_arr)}  — one entry per unique species.
    x_arr is in the same x-axis units as the plot.
    """
    signal_cols = [col for col in nmr_data if not col.startswith("_")]
    if not signal_cols:
        return {}

    all_sp       = network["all_species"]
    stoich       = network["stoich"]
    primary_sp   = params.get("primary_component", "")
    G0           = params["conc0"].get(primary_sp, 1.0)   # mM
    V0           = params["V0_mL"]
    is_solid     = params.get("titrant_is_solid", False)

    # ── Build (col, sp_name, n_H, raw_I, v_add_mL) list ─────────────────────
    # Species name extraction: a column header may be a real species (e.g. "GH"),
    # or a $variable name (e.g. "S012"), or a pandas-deduplicated duplicate of
    # either ("GH.1", "S012.2", …).  Strip the trailing ".N"/"_N" suffix first;
    # accept the result if it matches a real species OR a known $variable.
    # If neither, fall back to the column header as a last resort.
    variables_parsed = parsed.get("variables", {}) or {}
    entries = []
    for idx, col in enumerate(signal_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_sp and sp not in variables_parsed:
            sp = col.split(".")[0]
        if sp not in all_sp and sp not in variables_parsed:
            sp = col
        n_H    = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0
        raw_I  = nmr_data[col]["y"]
        v_add  = nmr_data[col]["v_add_mL"]   # mL (liquid) or x-axis value (solid)
        x_exp  = convert_exp_x(v_add, x_expr, parsed, params, network,
                                x_col_header=x_col_header)
        entries.append((col, sp, n_H, raw_I, v_add, x_exp))

    # ── Average I/n_H across duplicate signals of each species ───────────────
    # Use the x_arr from the first signal of each species (all should be same)
    sp_ratio  = {}   # {sp: ratio_array}   ratio = mean(I/n_H)
    sp_x      = {}   # {sp: x_array}
    sp_v_add  = {}   # {sp: v_add_array}
    for col, sp, n_H, raw_I, v_add, x_exp in entries:
        r = raw_I / n_H
        if sp not in sp_ratio:
            sp_ratio[sp] = []
            sp_x[sp]     = x_exp
            sp_v_add[sp] = v_add
        sp_ratio[sp].append(r)

    for sp in sp_ratio:
        sp_ratio[sp] = np.mean(np.vstack(sp_ratio[sp]), axis=0)

    # ── G-containing species and their stoichiometric coefficients ───────────
    # A column may name a real species (e.g. GH3) OR a $variables expression
    # (e.g. G012 = G + GH + GH2).  For variables we expand the expression and
    # sum the stoich values of the member species — in practice each member
    # contains exactly 1 primary component, so stoich_eff = 1.
    variables_parsed = parsed.get("variables", {})

    def _effective_stoich(col_sp):
        """Return effective stoich of col_sp w.r.t. primary_sp."""
        # Direct species lookup
        s = stoich.get((primary_sp, col_sp), 0)
        if s > 0:
            return s
        # Variable: expand to member species and average their stoich
        if col_sp in variables_parsed:
            expr = variables_parsed[col_sp]
            members = [p.strip() for p in expr.split("+") if p.strip()]
            total = 0.0; count = 0
            for m in members:
                # Handle "2 * GH2" style coefficients
                import re as _re
                mm = _re.match(r"^(\d+(?:\.\d+)?)\s*\*?\s*(\S+)$", m)
                if mm:
                    coeff_m, sp_m = float(mm.group(1)), mm.group(2)
                else:
                    coeff_m, sp_m = 1.0, m
                s_m = stoich.get((primary_sp, sp_m), 0)
                if s_m > 0:
                    total += coeff_m * s_m; count += coeff_m
            # effective stoich = total G atoms per molecule summed over members /
            # total molecule count (= 1 when all members have stoich 1 for G)
            if count > 0:
                return total / count
        return 0.0

    g_sp_stoich = {sp: _effective_stoich(sp)
                   for sp in sp_ratio
                   if _effective_stoich(sp) > 0}

    if not g_sp_stoich:
        # fallback: use primary species itself only
        g_sp_stoich = {primary_sp: 1}

    # ── Compute [G](x) and [Sp](x) for each x point ──────────────────────────
    # Use the x-grid of the first signal of any G-containing species
    ref_sp = next(iter(g_sp_stoich))
    x_ref_arr = sp_x[ref_sp]       # x-axis (equiv) — same units as theoretical curve
    v_add_ref = sp_v_add[ref_sp]   # raw column A values: mL (liquid) or equiv (solid)

    if is_solid:
        # Solid mode: volume is constant at V0, so [G_total] = G0 (no dilution).
        # v_add_ref here contains x-axis (equiv) values — used only for array shape.
        G_total = np.full_like(x_ref_arr, G0, dtype=float)
    else:
        # Liquid mode: v_add_ref is volume of titrant added in mL.
        # G is diluted as V increases: [G_total](x) = G0 * V0 / (V0 + V_add)
        G_total = G0 * V0 / np.maximum(V0 + v_add_ref, 1e-12)

    # denom(x) = Σ stoich(G,sp) × ratio_sp(x)   over G-containing species
    # Interpolate each species ratio onto the shared x-axis grid (x_ref_arr).
    # For both liquid and solid modes, sp_x values are in x-axis (equiv) units.
    denom = np.zeros_like(G_total)
    for sp, stoich_coeff in g_sp_stoich.items():
        ratio_on_ref = np.interp(x_ref_arr, sp_x[sp], sp_ratio[sp])
        denom += stoich_coeff * ratio_on_ref
    denom = np.maximum(denom, 1e-20)

    G_conc = G_total / denom   # mM

    # [Sp](x) = ratio_sp(x) × G_conc(x)  for ALL species (G-containing or not)
    bc = {}
    for sp, ratio_arr in sp_ratio.items():
        ratio_on_ref = np.interp(x_ref_arr, sp_x[sp], ratio_arr)
        c_bc = np.clip(ratio_on_ref * G_conc, 0.0, None)
        bc[sp] = (x_ref_arr, c_bc)   # (x_arr in equiv units, concentration_mM)

    return bc


def fit_nmr_integration(parsed: dict, network: dict, nmr_data: dict,
                        params: dict, logK_vals: dict, fit_keys: list,
                        x_expr: str, tolerance: float, maxiter: int,
                        timeout_s: float = 30.0, constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None,
                   *, compute_hessian: bool = True):
    """
    Fit equilibrium constants to slow-exchange NMR integration data.

    Physical model (per-spectrum normalisation):
        Each NMR spectrum is normalised to the primary-component signal = 1.
        Absolute concentrations are recovered via the G mass-balance:

            [Sp](x) = ratio_sp(x) × [G](x)
            [G](x)  = [G_total](x) / Σ stoich(G,sp)·ratio_sp(x)

        where ratio_sp = mean(I_sp/n_H_sp) over duplicate signals and
        [G_total](x) = G0·V0/(V0+V_add)  (liquid) or G0  (solid).

    The back-calculated concentrations are K-independent and computed once.
    The optimizer only moves K until the theoretical curves match them.
    """
    from scipy.optimize import minimize
    nmr_cfg  = parsed["nmr"]
    n_H_list = nmr_cfg.get("n_H_list", [])

    signal_cols = [col for col in nmr_data if not col.startswith("_")]
    if not signal_cols:
        return False, {}, {}, "No integration data columns found"

    # ── Solid mode: retrieve column-A header ─────────────────────────────────
    _x_col_header = nmr_data.get("_x_col_header", "")

    # ── Back-calculate concentrations from integrals (K-independent) ─────────
    bc = _nmr_integration_backCalc(nmr_data, n_H_list, params, network, x_expr, parsed,
                                   x_col_header=_x_col_header)
    if not bc:
        return False, {}, {}, "Back-calculation failed"

    # Build col→sp and col→n_H maps for statistics output
    # (Mirrors the species-name extraction in _nmr_integration_backCalc:
    # accept stripped name if it's a real species OR a known $variable.)
    all_sp   = network["all_species"]
    _vars    = parsed.get("variables", {}) or {}
    col_to_sp  = {}
    col_to_nH  = {}
    for idx, col in enumerate(signal_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_sp and sp not in _vars: sp = col.split(".")[0]
        if sp not in all_sp and sp not in _vars: sp = col
        col_to_sp[col] = sp
        col_to_nH[col] = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0

    # Unique species that have back-calculated data
    bc_species = list(bc.keys())


    # ── Concentration fitting setup ──────────────────────────────────────────
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)
    _CONC_MIN = 0.0

    _root_to_cname_nmr = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_nmr[_r] = _cn

    def _cb_nmr(root, default_val, sv_override=None):
        cname = _root_to_cname_nmr.get(root, root)
        sv  = sv_override if sv_override is not None else \
              parsed.get("concentrations", {}).get(cname, default_val)
        lo, hi = parsed.get("conc_bounds", {}).get(cname,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    def _tb_nmr(tkey, default_val, sv_override=None):
        sv  = sv_override if sv_override is not None else \
              parsed.get("titrant", {}).get(tkey, default_val)
        lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    # Seed from script values so optimizer starts inside bounds regardless of sidebar
    _root_to_cname_x0 = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_x0[_r] = _cn
    def _script_val(root):
        cname = _root_to_cname_x0.get(root, root)
        sv = parsed.get("concentrations", {}).get(cname)
        return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
    _x0_conc_nmr    = [_script_val(cn) for cn in fit_conc_keys]
    _x0_titrant_nmr = []
    for _tk in fit_titrant_keys:
        _tfree = _tk[:-1] if (_tk.endswith("t") or _tk.endswith("0")) else _tk
        _x0_titrant_nmr.append(params["titrant_mMs"].get(_tfree, 10.0))
    _bounds_conc_nmr    = [_cb_nmr(cn, _x0_conc_nmr[i])    for i, cn in enumerate(fit_conc_keys)]
    _bounds_titrant_nmr = [_tb_nmr(tk, _x0_titrant_nmr[i]) for i, tk in enumerate(fit_titrant_keys)]

    # Per-K soft bounds from the script's 'from X to Y' syntax (defaults if absent).
    # Honored as a quadratic penalty in objective() since Nelder-Mead ignores `bounds`.
    _DEFAULT_LO_K, _DEFAULT_HI_K = -2.0, 14.0
    _eq_by_kname_nmr = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
    def _resolve_k_bound_nmr(k, attr, default):
        v = _eq_by_kname_nmr.get(k, {}).get(attr)
        return float(v) if v is not None else float(default)
    _k_lo_nmr = np.array([_resolve_k_bound_nmr(k, "logK_lo", _DEFAULT_LO_K) for k in fit_keys])
    _k_hi_nmr = np.array([_resolve_k_bound_nmr(k, "logK_hi", _DEFAULT_HI_K) for k in fit_keys])

    def _update_bds_nmr(fitted_x):
        """Recentre bounds to ±20% of current fitted values after each pass."""
        for _i, _cn in enumerate(fit_conc_keys):
            _sv = float(np.clip(fitted_x[_n_k + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_conc_nmr[_i] = _cb_nmr(_cn, _sv, sv_override=_sv)
        for _i, _tk in enumerate(fit_titrant_keys):
            _sv = float(np.clip(fitted_x[_n_k + _n_c + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_titrant_nmr[_i] = _tb_nmr(_tk, _sv, sv_override=_sv)

    def _make_simplex_steps_nmr():
        """Step = full feasible range (hi - lo) per concentration/titrant dim."""
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else 1.5))
        c_steps = np.array([_bounds_conc_nmr[i][1]    - _bounds_conc_nmr[i][0]
                            for i in range(_n_c)])
        t_steps = np.array([_bounds_titrant_nmr[i][1] - _bounds_titrant_nmr[i][0]
                            for i in range(len(fit_titrant_keys))])
        return np.concatenate([k_steps, c_steps, t_steps]) \
               if len(c_steps) or len(t_steps) else k_steps

    def _make_params_nmr(trial):
        if not fitting_concs:
            return params
        n_k = len(fit_keys)
        p = dict(params)
        p["conc0"] = dict(params["conc0"])
        p["titrant_mMs"] = dict(params.get("titrant_mMs", {}))
        for i, cn in enumerate(fit_conc_keys):
            p["conc0"][cn] = float(np.clip(trial[n_k + i],
                                           _bounds_conc_nmr[i][0],
                                           _bounds_conc_nmr[i][1]))
        for i, tk in enumerate(fit_titrant_keys):
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            p["titrant_mMs"][tfree] = float(np.clip(trial[n_k + len(fit_conc_keys) + i],
                                                     _bounds_titrant_nmr[i][0],
                                                     _bounds_titrant_nmr[i][1]))
        return p

    def _simulate(logk_trial, _params=None):
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_curve(parsed, network, lk, _params or params)
        except Exception:
            return None

    def _get_bc(_p):
        """Return back-calculated concentrations for given params.
        Recomputed when fitting concentrations (G0 changes bc); cached otherwise."""
        if fitting_concs:
            return _nmr_integration_backCalc(nmr_data, n_H_list, _p, network, x_expr, parsed) or bc
        return bc

    def objective(trial, bc_frozen):
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        # Penalty for log-K outside per-K bounds
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None: return 1e12
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        ssr = 0.0
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            ssr += float(np.sum((c_bc - c_th) ** 2))
        cp = constraints_penalty(constraints or [], lk, ssr_scale=ssr)
        return ssr + cp + conc_penalty

    def data_objective(full_trial, bc_frozen):
        """Data-only SSR for Hessian."""
        _p = _make_params_nmr(full_trial) if fitting_concs else params
        c = _simulate(full_trial[:len(fit_keys)], _p)
        if c is None: return 1e12
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        ssr = 0.0
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            ssr += float(np.sum((c_bc - c_th) ** 2))
        return ssr

    import time
    x0  = np.concatenate([
        [logK_vals[k] for k in fit_keys],
        _x0_conc_nmr,
        _x0_titrant_nmr,
    ])
    n_p  = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _n_k = len(fit_keys)
    _n_c = len(fit_conc_keys)

    # ── Global timer starts here — covers Phase 0 AND the re-fit loop ───
    _global_start  = time.time()

    # ── Phase 0: warm-start logK before loop ────────────────────────────
    if fitting_concs and _n_k > 0:
        _bc_ph0 = _get_bc(_make_params_nmr(x0) if fitting_concs else params)
        _sp0 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                          for i in range(_n_k)])
        class _TimeoutP0(Exception): pass
        def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s, _bc=_bc_ph0):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP0()
            return objective(np.concatenate([lk, x0[_n_k:]]), _bc)
        try:
            _r0 = minimize(_ph0_obj, x0[:_n_k], method="Nelder-Mead",
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp0})
            x0 = np.concatenate([_r0.x, x0[_n_k:]])
        except (_TimeoutP0, Exception):
            pass

    # ── Re-fit loop ──────────────────────────────────────────────────────
    n_passes       = 200 if fitting_concs else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True; break

        _p_x0    = _make_params_nmr(x0) if fitting_concs else params
        bc_pass  = _get_bc(_p_x0)
        _base_steps = _make_simplex_steps_nmr()
        if _pass > 0 and fitting_concs:
            _base_steps[:_n_k] = 0.1
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])
        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0}

        def _obj_timed(trial, _tracker=_bt, _bc=bc_pass,
                        _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            f = objective(trial, _bc)
            if f < _tracker["f"]: _tracker["f"] = f; _tracker["x"] = trial.copy()
            if time.time() - _t0 > _tlim: raise _Timeout()
            return f

        try:
            result = minimize(_obj_timed, x0, method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult(); _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])

        if fitting_concs:
            x0_prev = x0.copy()
            x0 = result.x.copy(); best_x_global = x0.copy()
            _update_bds_nmr(x0)
            if _any_timed_out: break
            if np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:])) < tolerance: break
        else:
            best_x_global = result.x.copy()

    class _BestResult:
        x = best_x_global; success = getattr(result, "success", False)
        fun = getattr(result, "fun", np.inf); nit = _total_nit
    result = _BestResult()

    fitted_logKs = {fit_keys[i]: result.x[i] for i in range(len(fit_keys))}
    _fitted_concs_nmr    = {fit_conc_keys[i]: float(result.x[len(fit_keys) + i])
                            for i in range(len(fit_conc_keys))}
    _fitted_titrants_nmr = {fit_titrant_keys[i]: float(result.x[len(fit_keys) + len(fit_conc_keys) + i])
                             for i in range(len(fit_titrant_keys))}
    _params_final_integ = _make_params_nmr(result.x) if fitting_concs else params

    # ── Statistics ────────────────────────────────────────────────────────────
    c_final = _simulate(result.x[:len(fit_keys)], _params_final_integ)
    bc_final = _get_bc(_params_final_integ)
    all_res = []
    all_obs = []
    # v2: per-species arrays for bootstrap resampling
    per_col_y_obs   = {}
    per_col_y_calc  = {}
    per_col_x       = {}

    if c_final is not None:
        x_sim_f, _ = evaluate_x_expression(x_expr, c_final, parsed)
        for sp, (x_bc, c_bc) in bc_final.items():
            c_th = np.interp(x_bc, x_sim_f, _resolve_c(c_final, sp, parsed, x_sim_f))
            all_res.extend((c_bc - c_th).tolist())
            all_obs.extend(c_bc.tolist())
            per_col_y_obs[sp]  = np.asarray(c_bc, dtype=float)
            per_col_y_calc[sp] = np.asarray(c_th, dtype=float)
            per_col_x[sp]      = np.asarray(x_bc, dtype=float)

    residuals = np.array(all_res)
    y_obs_arr = np.array(all_obs)
    ssr  = float(np.sum(residuals ** 2))
    sst  = float(np.sum((y_obs_arr - y_obs_arr.mean()) ** 2)) if len(y_obs_arr) > 1 else 1.0
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

    # Parameter errors via finite-difference Hessian (skipped if compute_hessian=False)
    _bc_final_h = _get_bc(_params_final_integ)
    _n_total_params_integ = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    param_errors_integ = {}
    _cov_mat = None
    _cov_names_integ = list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(lambda pv: data_objective(pv, _bc_final_h),
                                   result.x, ssr, len(residuals), _n_total_params_integ)
        param_errors_integ = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors_integ[fit_conc_keys[_i]] = _err_idx[_idx]
        for _i in range(len(fit_titrant_keys)):
            _idx = len(fit_keys) + len(fit_conc_keys) + _i
            if _idx in _err_idx: param_errors_integ[fit_titrant_keys[_i]] = _err_idx[_idx]

    # sp_concs format: {sp: [(x_arr, c_bc_arr), ...]}  (list for compatibility)
    sp_concs = {sp: [(x_bc, c_bc)] for sp, (x_bc, c_bc) in bc_final.items()}

    stats = {
        "r_squared":    r2,
        "rmse":         rmse,
        "ssr":          ssr,
        "n_points":     len(residuals),
        "n_params":     len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys),
        "param_values": fitted_logKs,
        "param_errors": param_errors_integ,
        "param_cov":       _cov_mat,
        "param_cov_names": _cov_names_integ,
        "sp_concs":     sp_concs,
        "col_to_sp":    col_to_sp,
        "col_to_nH":    col_to_nH,
        "fit_mode":     "integration",
        "fitted_concs":    _fitted_concs_nmr,
        "fitted_titrants": _fitted_titrants_nmr,
        "n_iter":       getattr(result, "nit", 0),
        "timed_out":    not getattr(result, "success", True),
        "pure_shifts":     {},   # not used in integration mode
        "delta_vecs_all":  {},
        "delta_bound_all": {},
        "delta_free":      {},
        "x_free_val":      {},
        "col_to_target":   {},
        "ref_corrections": {},
        # v2 diagnostics — flat residual arrays
        "y_obs":           np.asarray(y_obs_arr,             dtype=float),
        "y_calc":          np.asarray(y_obs_arr - residuals, dtype=float),
        "residuals":       np.asarray(residuals,             dtype=float),
        # v2 bootstrap — per-species arrays
        "per_col_y_obs":   per_col_y_obs,
        "per_col_y_calc":  per_col_y_calc,
        "per_col_x":       per_col_x,
    }
    _to   = stats.get("timed_out", False)
    _r2   = stats.get("r_squared", 0.0)
    _conv = result.success or ssr < 1e-6 or (_to and _r2 >= 0.99)
    if _to and _r2 >= 0.99:
        stats["timed_out"] = False
    _msg = "NMR integration fit complete"
    return _conv, fitted_logKs, stats, _msg


def fit_nmr_mixed(parsed: dict, network: dict, nmr_data: dict,
                  params: dict, logK_vals: dict, fit_keys: list,
                  x_expr: str, tolerance: float, maxiter: int,
                  timeout_s: float = 30.0, constraints=None,
                   fit_conc_keys=None, fit_titrant_keys=None,
                   *, compute_hessian: bool = True):
    """
    Fit equilibrium constants to mixed slow+fast exchange NMR data.

    The $nmr section contains BOTH lines:
        integration: n1, n2, ...   <- first n_integ data columns are slow-exchange peaks
        shift: VarName             <- remaining columns are fast-exchange chemical shifts

    Column layout in the Excel file:
        [V_add] | [integ col 1] ... [integ col n_integ] | [shift col 1] ...

    Integration columns:
        Back-calculated to absolute concentrations via the G mass-balance
        (same as pure integration mode).  These are K-independent and computed once.

    Shift columns:
        Each belongs to the fast-exchange variable named in `shift:` (e.g. G012).
        δ_obs = Σᵢ [Spᵢ]·δᵢ / Σᵢ [Spᵢ]   where the sum runs over member species.
        Δδ values are solved analytically (linear least squares) inside the optimizer.

    Combined objective:
        SSR = Σ (c_bc - c_theory)²   [integration part, in mM²]
            + w · Σ (Δδ_obs - Δδ_calc)²   [shift part, in ppm²]
        where w scales the shift residuals to comparable magnitude.
    """
    from scipy.optimize import minimize

    nmr_cfg  = parsed["nmr"]
    n_H_list = nmr_cfg.get("n_H_list", [])
    n_integ  = nmr_cfg.get("n_integ", len(n_H_list))
    shift_targets = nmr_cfg.get("targets", [])
    _shift_constraints = parsed.get("shift_constraints", []) or []

    # ── Solid mode: retrieve column-A header ─────────────────────────────────
    _x_col_header = nmr_data.get("_x_col_header", "")

    # ── Split nmr_data into integration and shift columns ────────────────────
    all_cols   = [col for col in nmr_data if not col.startswith("_")]
    integ_cols = all_cols[:n_integ]
    shift_cols = all_cols[n_integ:]

    integ_data = {col: nmr_data[col] for col in integ_cols}
    shift_data = {col: nmr_data[col] for col in shift_cols}

    if not integ_cols and not shift_cols:
        return False, {}, {}, "No NMR data columns found"

    # ── Per-column shift bounds (the 'shift: -0.8, free, 1.0, ...' syntax) ──
    _per_col_bounds = get_per_column_bounds(_shift_constraints)
    _col_to_bound = {}
    if _per_col_bounds is not None:
        if len(_per_col_bounds) != len(shift_cols):
            return False, {}, {}, (
                f"Per-column shift bounds list has {len(_per_col_bounds)} entries "
                f"but the data has {len(shift_cols)} shift columns. "
                f"Use 'free' for columns you don't want to constrain."
            )
        _col_to_bound = dict(zip(shift_cols, _per_col_bounds))

    # ── Back-calculate concentrations from integration columns (K-independent) ─
    bc = {}
    if integ_cols:
        bc = _nmr_integration_backCalc(integ_data, n_H_list[:n_integ],
                                        params, network, x_expr, parsed,
                                        x_col_header=_x_col_header)

    # ── Prepare shift data structures (same as fit_nmr_shifts) ───────────────
    col_to_target = {}
    for col in shift_cols:
        for tgt in shift_targets:
            if col == tgt or col.startswith(tgt + ".") or col.startswith(tgt + "_"):
                col_to_target[col] = tgt; break
        else:
            col_to_target[col] = shift_targets[0] if shift_targets else col

    # δ_free: first data point per shift column
    delta_free  = {}
    x_free_val  = {}
    for col in shift_cols:
        delta_free[col] = float(nmr_data[col]["y"][0])
        x_arr = convert_exp_x(nmr_data[col]["v_add_mL"], x_expr, parsed, params, network,
                               x_col_header=_x_col_header)
        x_free_val[col] = float(x_arr[0])

    # ── Pinned intrinsic shifts ($nmr 'read:' + sheet-2 known shifts) ─────
    _read_species  = list(nmr_cfg.get("read", []) or [])
    _known_shifts  = nmr_data.get("_known_shifts", {}) or {}
    _col_to_pinned = {}
    if _read_species:
        for sp in _read_species:
            if sp not in _known_shifts:
                return False, {}, {}, (
                    f"$nmr 'read: {sp}' but species '{sp}' not found in sheet 2 of "
                    f"the data file (expected row with '{sp}' in column A and "
                    f"{len(shift_cols)} chemical-shift values to its right)."
                )
            vals = _known_shifts[sp]
            if len(vals) != len(shift_cols):
                return False, {}, {}, (
                    f"$nmr 'read: {sp}': sheet 2 has {len(vals)} values for '{sp}' "
                    f"but the data has {len(shift_cols)} shift columns. "
                    f"Provide one value per shift column, in the same order."
                )
            for k, col in enumerate(shift_cols):
                dd_val = float(vals[k]) - float(delta_free[col])
                _col_to_pinned.setdefault(col, {})[sp] = dd_val

    # noref flag: when True, dd_ref is NOT auto-pinned to 0; instead at least
    # one read: anchor is required to fix the absolute scale.  Default False
    # preserves legacy "Δδ relative to ref" behavior so tutorials run unchanged.
    _nmr_noref = bool(nmr_cfg.get("noref", False))


    # ── Concentration fitting setup ──────────────────────────────────────────
    fit_conc_keys    = list(fit_conc_keys    or [])
    fit_titrant_keys = list(fit_titrant_keys or [])
    fitting_concs    = bool(fit_conc_keys or fit_titrant_keys)
    _CONC_MIN = 0.0

    _root_to_cname_nmr = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_nmr[_r] = _cn

    def _cb_nmr(root, default_val, sv_override=None):
        cname = _root_to_cname_nmr.get(root, root)
        sv  = sv_override if sv_override is not None else \
              parsed.get("concentrations", {}).get(cname, default_val)
        lo, hi = parsed.get("conc_bounds", {}).get(cname,
                 parsed.get("conc_bounds", {}).get(root, (None, None)))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    def _tb_nmr(tkey, default_val, sv_override=None):
        sv  = sv_override if sv_override is not None else \
              parsed.get("titrant", {}).get(tkey, default_val)
        lo, hi = parsed.get("titrant_bounds", {}).get(tkey, (None, None))
        lo = max(_CONC_MIN, lo) if lo is not None else max(_CONC_MIN, sv * 0.80)
        hi = hi if hi is not None else sv * 1.20
        return (lo, hi)

    # Seed from script values so optimizer starts inside bounds regardless of sidebar
    _root_to_cname_x0 = {}
    for _cn in parsed.get("concentrations", {}):
        _r = _cn[:-1] if _cn.endswith("0") else _cn
        _root_to_cname_x0[_r] = _cn
    def _script_val(root):
        cname = _root_to_cname_x0.get(root, root)
        sv = parsed.get("concentrations", {}).get(cname)
        return float(sv) if sv is not None else float(params["conc0"].get(root, 1.0))
    _x0_conc_nmr    = [_script_val(cn) for cn in fit_conc_keys]
    _x0_titrant_nmr = []
    for _tk in fit_titrant_keys:
        _tfree = _tk[:-1] if (_tk.endswith("t") or _tk.endswith("0")) else _tk
        _x0_titrant_nmr.append(params["titrant_mMs"].get(_tfree, 10.0))
    _bounds_conc_nmr    = [_cb_nmr(cn, _x0_conc_nmr[i])    for i, cn in enumerate(fit_conc_keys)]
    _bounds_titrant_nmr = [_tb_nmr(tk, _x0_titrant_nmr[i]) for i, tk in enumerate(fit_titrant_keys)]

    # Per-K soft bounds from the script's 'from X to Y' syntax (defaults if absent).
    # Honored as a quadratic penalty in objective() since Nelder-Mead ignores `bounds`.
    _DEFAULT_LO_K, _DEFAULT_HI_K = -2.0, 14.0
    _eq_by_kname_nmr = {eq["kname"]: eq for eq in parsed.get("equilibria", [])}
    def _resolve_k_bound_nmr(k, attr, default):
        v = _eq_by_kname_nmr.get(k, {}).get(attr)
        return float(v) if v is not None else float(default)
    _k_lo_nmr = np.array([_resolve_k_bound_nmr(k, "logK_lo", _DEFAULT_LO_K) for k in fit_keys])
    _k_hi_nmr = np.array([_resolve_k_bound_nmr(k, "logK_hi", _DEFAULT_HI_K) for k in fit_keys])

    def _update_bds_nmr(fitted_x):
        """Recentre bounds to ±20% of current fitted values after each pass."""
        for _i, _cn in enumerate(fit_conc_keys):
            _sv = float(np.clip(fitted_x[_n_k + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_conc_nmr[_i] = _cb_nmr(_cn, _sv, sv_override=_sv)
        for _i, _tk in enumerate(fit_titrant_keys):
            _sv = float(np.clip(fitted_x[_n_k + _n_c + _i], _CONC_MIN + 1e-12, 1e9))
            _bounds_titrant_nmr[_i] = _tb_nmr(_tk, _sv, sv_override=_sv)

    def _make_simplex_steps_nmr():
        """Step = full feasible range (hi - lo) per concentration/titrant dim."""
        k_steps = np.full(_n_k, (1e-9 if maxiter <= 1 else 1.5))
        c_steps = np.array([_bounds_conc_nmr[i][1]    - _bounds_conc_nmr[i][0]
                            for i in range(_n_c)])
        t_steps = np.array([_bounds_titrant_nmr[i][1] - _bounds_titrant_nmr[i][0]
                            for i in range(len(fit_titrant_keys))])
        return np.concatenate([k_steps, c_steps, t_steps]) \
               if len(c_steps) or len(t_steps) else k_steps

    def _make_params_nmr(trial):
        if not fitting_concs:
            return params
        n_k = len(fit_keys)
        p = dict(params)
        p["conc0"] = dict(params["conc0"])
        p["titrant_mMs"] = dict(params.get("titrant_mMs", {}))
        for i, cn in enumerate(fit_conc_keys):
            p["conc0"][cn] = float(np.clip(trial[n_k + i],
                                           _bounds_conc_nmr[i][0],
                                           _bounds_conc_nmr[i][1]))
        for i, tk in enumerate(fit_titrant_keys):
            tfree = tk[:-1] if (tk.endswith("t") or tk.endswith("0")) else tk
            p["titrant_mMs"][tfree] = float(np.clip(trial[n_k + len(fit_conc_keys) + i],
                                                     _bounds_titrant_nmr[i][0],
                                                     _bounds_titrant_nmr[i][1]))
        return p

    def _simulate(logk_trial, _params=None):
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_curve(parsed, network, lk, _params or params)
        except Exception:
            return None

    def _x_for_col(col):
        return convert_exp_x(nmr_data[col]["v_add_mL"], x_expr, parsed, params, network,
                             x_col_header=_x_col_header)

    def _shift_ssr(c):
        """Compute shift SSR with analytic Δδ for current curve c."""
        total = 0.0
        for col in shift_cols:
            tgt       = col_to_target.get(col, shift_targets[0] if shift_targets else col)
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
            x_exp     = _x_for_col(col)
            x_free_c  = x_free_val[col]

            # Weighted denominator
            denom_full = np.maximum(
                sum(coeff * c.get(sp, np.zeros_like(x_sim))
                    for coeff, sp in sp_coeffs), 1e-20)
            denom_ref  = max(sum(
                coeff * float(np.interp(x_free_c, x_sim, c.get(sp, np.zeros_like(x_sim))))
                for coeff, sp in sp_coeffs), 1e-20)

            # Build full design matrix (all species columns including ref)
            n_pts    = len(x_exp)
            all_target_sp = list(sp_coeffs)
            X = np.zeros((n_pts, len(all_target_sp)))
            for i, (coeff, sp) in enumerate(all_target_sp):
                F_full = coeff * np.interp(x_exp, x_sim, c.get(sp, np.zeros_like(x_sim))) /                          np.maximum(np.interp(x_exp, x_sim, denom_full), 1e-20)
                F_ref  = coeff * float(np.interp(x_free_c, x_sim,
                         c.get(sp, np.zeros_like(x_sim)))) / denom_ref
                X[:, i] = F_full - F_ref

            dobs_rel = nmr_data[col]["y"] - delta_free[col]
            _col_bd = _col_to_bound.get(col)
            _col_pn = _col_to_pinned.get(col)
            if X.shape[1] > 0 and np.any(np.abs(X) > 1e-15):
                try:
                    dd, calc, _ssr = solve_shifts(
                        X, dobs_rel,
                        [sp for _, sp in all_target_sp], sp_coeffs[0][1],
                        _shift_constraints,
                        column_bound=_col_bd, pinned_dd=_col_pn,
                        noref=_nmr_noref,
                    )
                    residuals = dobs_rel - calc
                except ValueError:
                    dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
                    residuals = dobs_rel - X @ dd
            else:
                residuals = dobs_rel
            total += float(np.sum(residuals**2))
        return total

    def _get_bc_mixed(_p):
        """Recompute bc when fitting concentrations (G0 changes bc); cached otherwise."""
        if fitting_concs and integ_cols:
            return _nmr_integration_backCalc(integ_data, n_H_list[:n_integ],
                                             _p, network, x_expr, parsed) or bc
        return bc

    def objective(trial, bc_frozen):
        logk_trial = trial[:len(fit_keys)]
        lk = logK_vals.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        conc_penalty = 0.0
        # Penalty for log-K outside per-K bounds
        for kv, _lo, _hi in zip(logk_trial, _k_lo_nmr, _k_hi_nmr):
            if kv < _lo: conc_penalty += 1e6 * (kv - _lo) ** 2
            elif kv > _hi: conc_penalty += 1e6 * (kv - _hi) ** 2
        if fitting_concs:
            n_k = len(fit_keys)
            for i in range(len(fit_conc_keys)):
                v = trial[n_k + i]; lo, hi = _bounds_conc_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
            for i in range(len(fit_titrant_keys)):
                v = trial[n_k + len(fit_conc_keys) + i]; lo, hi = _bounds_titrant_nmr[i]
                if v < lo: conc_penalty += 1e6*(v-lo)**2
                elif v > hi: conc_penalty += 1e6*(v-hi)**2
        _p = _make_params_nmr(trial) if fitting_concs else params
        c = _simulate(logk_trial, _p)
        if c is None: return 1e12
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        integ_ssr = 0.0
        integ_var = max(sum(float(np.var(c_bc)) for _, (_, c_bc) in bc_frozen.items()), 1e-20)
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            integ_ssr += float(np.sum((c_bc - c_th)**2)) / integ_var
        shift_ssr_raw = _shift_ssr(c)
        shift_var = max(sum(float(np.var(nmr_data[col2]["y"]))
                            for col2 in shift_cols), 1e-20) if shift_cols else 1.0
        total_ssr = integ_ssr + shift_ssr_raw / shift_var
        cp = constraints_penalty(constraints or [], lk, ssr_scale=total_ssr)
        return total_ssr + cp + conc_penalty

    def data_objective(full_trial, bc_frozen):
        """Data-only SSR for Hessian."""
        _p = _make_params_nmr(full_trial) if fitting_concs else params
        c = _simulate(full_trial[:len(fit_keys)], _p)
        if c is None: return 1e12
        x_sim, _ = evaluate_x_expression(x_expr, c, parsed)
        integ_ssr = 0.0
        integ_var = max(sum(float(np.var(c_bc)) for _, (_, c_bc) in bc_frozen.items()), 1e-20)
        for sp, (x_bc, c_bc) in bc_frozen.items():
            c_th = np.interp(x_bc, x_sim, _resolve_c(c, sp, parsed, x_sim))
            integ_ssr += float(np.sum((c_bc - c_th)**2)) / integ_var
        shift_ssr_raw = _shift_ssr(c)
        shift_var = max(sum(float(np.var(nmr_data[col2]["y"]))
                            for col2 in shift_cols), 1e-20) if shift_cols else 1.0
        return integ_ssr + shift_ssr_raw / shift_var

    import time

    x0 = np.concatenate([
        [logK_vals[k] for k in fit_keys],
        _x0_conc_nmr,
        _x0_titrant_nmr,
    ])
    n_p  = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _n_k = len(fit_keys)
    _n_c = len(fit_conc_keys)

    # ── Global timer starts here — covers Phase 0 AND the re-fit loop ───
    _global_start  = time.time()

    # ── Phase 0: warm-start logK before loop ────────────────────────────
    if fitting_concs and _n_k > 0:
        _bc_ph0 = _get_bc_mixed(_make_params_nmr(x0) if fitting_concs else params)
        _sp0 = np.vstack([x0[:_n_k]] + [x0[:_n_k] + np.eye(_n_k)[i]*1.5
                                          for i in range(_n_k)])
        class _TimeoutP0(Exception): pass
        def _ph0_obj(lk, _t0=_global_start, _tlim=timeout_s, _bc=_bc_ph0):
            if time.time() - _t0 > _tlim * 0.4:
                raise _TimeoutP0()
            return objective(np.concatenate([lk, x0[_n_k:]]), _bc)
        try:
            _r0 = minimize(_ph0_obj, x0[:_n_k], method="Nelder-Mead",
                           options={"maxiter": maxiter//10, "xatol": tolerance,
                                    "fatol": tolerance * 1e-4, "adaptive": True,
                                    "initial_simplex": _sp0})
            x0 = np.concatenate([_r0.x, x0[_n_k:]])
        except (_TimeoutP0, Exception):
            pass

    # ── Re-fit loop ──────────────────────────────────────────────────────
    n_passes       = 200 if fitting_concs else 1
    best_x_global  = x0.copy()
    _total_nit     = 0
    _any_timed_out = False
    result         = None

    for _pass in range(n_passes):
        _elapsed   = time.time() - _global_start
        _remaining = timeout_s - _elapsed
        if _remaining <= 0.5:
            _any_timed_out = True; break

        _p_x0    = _make_params_nmr(x0) if fitting_concs else params
        bc_pass  = _get_bc_mixed(_p_x0)
        _base_steps = _make_simplex_steps_nmr()
        if _pass > 0 and fitting_concs:
            _base_steps[:_n_k] = 0.1
        _pass_steps  = _base_steps
        init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * _pass_steps[i]
                                         for i in range(n_p)])
        class _Timeout(Exception): pass
        _bt = {"x": x0.copy(), "f": np.inf, "start": time.time(), "nit": 0}

        def _obj_timed(trial, _tracker=_bt, _bc=bc_pass,
                        _t0=_global_start, _tlim=timeout_s):
            _tracker["nit"] += 1
            f = objective(trial, _bc)
            if f < _tracker["f"]: _tracker["f"] = f; _tracker["x"] = trial.copy()
            if time.time() - _t0 > _tlim: raise _Timeout()
            return f

        try:
            result = minimize(_obj_timed, x0, method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x = _bt["x"]; success = False; fun = _bt["f"]; nit = _bt["nit"]
            result = _MockResult(); _any_timed_out = True

        _total_nit += getattr(result, "nit", _bt["nit"])

        if fitting_concs:
            x0_prev = x0.copy()
            x0 = result.x.copy(); best_x_global = x0.copy()
            _update_bds_nmr(x0)
            if _any_timed_out: break
            if np.max(np.abs(x0[_n_k:] - x0_prev[_n_k:])) < tolerance: break
        else:
            best_x_global = result.x.copy()

    class _BestResult:
        x = best_x_global; success = getattr(result, "success", False)
        fun = getattr(result, "fun", np.inf); nit = _total_nit
    result = _BestResult()

    timed_out = _any_timed_out
    fitted_logKs = {fit_keys[i]: result.x[i] for i in range(len(fit_keys))}
    _fitted_concs_nmr    = {fit_conc_keys[i]: float(result.x[len(fit_keys) + i])
                            for i in range(len(fit_conc_keys))}
    _fitted_titrants_nmr = {fit_titrant_keys[i]: float(result.x[len(fit_keys) + len(fit_conc_keys) + i])
                             for i in range(len(fit_titrant_keys))}
    _params_final_mixed = _make_params_nmr(result.x) if fitting_concs else params

    # ── Final pass: collect all stats and shift Δδ vectors ──────────────────
    c_final = _simulate(result.x[:len(fit_keys)], _params_final_mixed)
    sp_concs = {sp: [(x_bc, c_bc)] for sp, (x_bc, c_bc) in _get_bc_mixed(_params_final_mixed).items()}

    # Re-run shift analytics to get delta_vecs_all, pure_shifts etc.
    # (reuse the machinery from fit_nmr_shifts)
    delta_vecs_all  = {}
    delta_bound_all = {}
    pure_shifts     = {}
    ref_corrections = {}
    # Track integration and shift residuals SEPARATELY to avoid unit mixing.
    # Integration residuals are in mM; shift residuals are in ppm (Δδ).
    # Mixing them into one array produces nonsensical R² and RMSE.
    integ_res = []; integ_obs = []
    shift_res = []; shift_obs = []   # stored as Δδ (relative), not absolute ppm
    # v2: per-source arrays for bootstrap (integration species + shift columns)
    per_col_y_obs_integ  = {}; per_col_y_calc_integ  = {}; per_col_x_integ  = {}
    per_col_y_obs_shift  = {}; per_col_y_calc_shift  = {}; per_col_x_shift  = {}

    if c_final is not None:
        x_sim_f, _ = evaluate_x_expression(x_expr, c_final, parsed)
        _bc_final_mixed = _get_bc_mixed(_params_final_mixed)

        # Integration residuals (mM)
        for sp, (x_bc, c_bc) in _bc_final_mixed.items():
            c_th = np.interp(x_bc, x_sim_f, _resolve_c(c_final, sp, parsed, x_sim_f))
            integ_res.extend((c_bc - c_th).tolist())
            integ_obs.extend(c_bc.tolist())
            per_col_y_obs_integ[sp]  = np.asarray(c_bc, dtype=float)
            per_col_y_calc_integ[sp] = np.asarray(c_th, dtype=float)
            per_col_x_integ[sp]      = np.asarray(x_bc, dtype=float)

        # Shift analytics (Δδ ppm)
        for col in shift_cols:
            tgt       = col_to_target.get(col, shift_targets[0] if shift_targets else col)
            sp_coeffs = _get_species_for_target(tgt, parsed, network)
            if not sp_coeffs: continue
            x_exp     = _x_for_col(col)
            x_free_c  = x_free_val[col]
            df0       = delta_free[col]

            denom_full = np.maximum(
                sum(coeff * c_final.get(sp, np.zeros_like(x_sim_f))
                    for coeff, sp in sp_coeffs), 1e-20)
            denom_ref  = max(sum(
                coeff * float(np.interp(x_free_c, x_sim_f, c_final.get(sp, np.zeros_like(x_sim_f))))
                for coeff, sp in sp_coeffs), 1e-20)

            # Build full design matrix (all species columns including ref)
            n_pts    = len(x_exp)
            all_target_sp = list(sp_coeffs)
            X = np.zeros((n_pts, len(all_target_sp)))
            for i, (coeff, sp) in enumerate(all_target_sp):
                F_full = coeff * np.interp(x_exp, x_sim_f, c_final.get(sp, np.zeros_like(x_sim_f))) /                          np.maximum(np.interp(x_exp, x_sim_f, denom_full), 1e-20)
                F_ref  = coeff * float(np.interp(x_free_c, x_sim_f,
                         c_final.get(sp, np.zeros_like(x_sim_f)))) / denom_ref
                X[:, i] = F_full - F_ref

            dobs_rel = nmr_data[col]["y"] - df0
            _col_bd = _col_to_bound.get(col)
            _col_pn = _col_to_pinned.get(col)
            if X.shape[1] > 0:
                try:
                    dd, calc_rel, _ = solve_shifts(
                        X, dobs_rel,
                        [sp for _, sp in all_target_sp], sp_coeffs[0][1],
                        _shift_constraints,
                        column_bound=_col_bd, pinned_dd=_col_pn,
                        noref=_nmr_noref,
                    )
                except ValueError:
                    dd = np.linalg.lstsq(X, dobs_rel, rcond=None)[0]
                    calc_rel = X @ dd
            else:
                dd = np.array([])
                calc_rel = np.zeros_like(dobs_rel)

            # Use Δδ (relative shifts) — NOT absolute ppm — for statistics
            shift_res.extend((dobs_rel - calc_rel).tolist())
            shift_obs.extend(dobs_rel.tolist())
            per_col_y_obs_shift[col]  = np.asarray(dobs_rel, dtype=float)
            per_col_y_calc_shift[col] = np.asarray(calc_rel, dtype=float)
            per_col_x_shift[col]      = np.asarray(x_exp,    dtype=float)

            # In the new convention dd[i] = δ_i_absolute − δ_obs(V=0).
            # Legacy (no noref): dd[ref] = 0 by auto-pin.  Noref: dd[ref] is fitted.
            sp_dd = {sp: float(dd[i]) if i < len(dd) else 0.0
                     for i, (_, sp) in enumerate(all_target_sp)}
            # Normalize when not anchored (see fit_nmr_shifts for rationale).
            if not (_nmr_noref and _read_species):
                ref_sp = sp_coeffs[0][1]
                ref_dd = sp_dd.get(ref_sp, 0.0)
                if abs(ref_dd) > 1e-12:
                    sp_dd = {k: v - ref_dd for k, v in sp_dd.items()}
            delta_vecs_all[col] = sp_dd
            if len(sp_coeffs) == 2:
                non_ref_sp = sp_coeffs[1][1]
                delta_bound_all[col] = sp_dd.get(non_ref_sp, 0.0)

            # Pure-species shifts: stored as dd values (= δ_absolute − δ_obs(V=0)).
            # Display/export adds δ_obs(V=0) back when the scale is anchored
            # (noref=True with at least one read: pin).
            # ref_corrections[col] is defined as Σᵢ Fᵢ(x_ref)·ddᵢ over ALL
            # species (incl. reference).  In noref+read mode dd[ref] may
            # be nonzero so the ref species term must be kept.
            ref_corrections[col] = sum(
                coeff * float(np.interp(x_free_c, x_sim_f,
                              c_final.get(sp, np.zeros_like(x_sim_f)))) / denom_ref
                * sp_dd.get(sp, 0.0)
                for coeff, sp in sp_coeffs   # all species, incl. ref
            )
            if tgt not in pure_shifts:
                pure_shifts[tgt] = {}
            pure_shifts[tgt][col] = dict(sp_dd)

    # ── Per-component statistics ─────────────────────────────────────────────
    def _r2_rmse(res_list, obs_list):
        if not res_list: return 0.0, 0.0, 0.0
        r = np.array(res_list); o = np.array(obs_list)
        ssr_ = float(np.sum(r**2))
        sst_ = float(np.sum((o - o.mean())**2)) if len(o) > 1 else 1.0
        return 1.0 - ssr_/max(sst_, 1e-30), float(np.sqrt(ssr_/max(len(r),1))), ssr_

    r2_integ, rmse_integ, ssr_integ = _r2_rmse(integ_res, integ_obs)
    r2_shift, rmse_shift, ssr_shift = _r2_rmse(shift_res, shift_obs)

    # Combined data point count
    n_integ_pts = len(integ_res)
    n_shift_pts = len(shift_res)
    n_total     = n_integ_pts + n_shift_pts

    # Overall R²: weighted average by data point count
    r2   = (r2_integ * n_integ_pts + r2_shift * n_shift_pts) / max(n_total, 1)
    ssr  = ssr_integ + ssr_shift
    rmse = float(np.sqrt(ssr / max(n_total, 1)))   # mixed units — shown separately

    # v2: combined flat residuals (integration first, then shift)
    _mixed_yobs = np.asarray(list(integ_obs) + list(shift_obs), dtype=float)
    _mixed_res  = np.asarray(list(integ_res) + list(shift_res), dtype=float)
    _mixed_ycalc = _mixed_yobs - _mixed_res

    # Parameter errors via finite-difference Hessian (skipped if compute_hessian=False).
    # For mixed mode the objective is dimensionless (normalized chi-squared), so
    # σ² is just 1/(n-p) — we're already working in units of variance.
    # Use data_objective (no penalty) so constraint curvature doesn't dominate.
    _n_total_params_mixed = len(fit_keys) + len(fit_conc_keys) + len(fit_titrant_keys)
    _bc_final_mixed_h = _get_bc_mixed(_params_final_mixed)
    param_errors_mixed = {}
    _cov_mat = None
    _cov_names_mixed = list(fit_keys) + list(fit_conc_keys) + list(fit_titrant_keys)
    if compute_hessian:
        _err_idx, _cov_mat = _hessian_errors(lambda pv: data_objective(pv, _bc_final_mixed_h),
                                   result.x, ssr, n_total, _n_total_params_mixed)
        param_errors_mixed = {fit_keys[i]: _err_idx[i] for i in range(len(fit_keys)) if i in _err_idx}
        for _i in range(len(fit_conc_keys)):
            _idx = len(fit_keys) + _i
            if _idx in _err_idx: param_errors_mixed[fit_conc_keys[_i]] = _err_idx[_idx]
        for _i in range(len(fit_titrant_keys)):
            _idx = len(fit_keys) + len(fit_conc_keys) + _i
            if _idx in _err_idx: param_errors_mixed[fit_titrant_keys[_i]] = _err_idx[_idx]

    # ── Variance-normalized chi²-RMSE for the LST ──────────────────────────
    # The optimizer minimized the variance-normalized objective
    # chi² = integ_ssr/integ_var + shift_ssr_raw/shift_var (mixed units
    # are made comparable by the per-component variance weights).  The
    # raw `rmse` reported above mixes mM² and ppm² and has its minimum
    # at a DIFFERENT point than the chi² minimum where the fit lives,
    # so the local-sensitivity test (which uses paired-difference
    # geometry that only makes sense at a local minimum of the probed
    # objective) needs the chi² value, not the raw rmse.  We expose
    # both: `rmse` stays raw for backward-compat / display, and
    # `chi2_rmse` is what the LST reads.
    try:
        _chi2_final = float(data_objective(result.x, _bc_final_mixed_h))
        _chi2_rmse  = float(np.sqrt(max(_chi2_final, 0.0) / max(n_total, 1)))
    except Exception:
        _chi2_rmse = float(rmse)  # fallback

    # Build col_to_sp and col_to_nH for display
    # (Mirrors the species-name extraction in _nmr_integration_backCalc:
    # accept stripped name if it's a real species OR a known $variable.)
    all_sp_net = network["all_species"]
    _vars      = parsed.get("variables", {}) or {}
    col_to_sp = {}; col_to_nH = {}
    for idx, col in enumerate(integ_cols):
        sp = re.split(r"[._]\d+$", col)[0]
        if sp not in all_sp_net and sp not in _vars: sp = col.split(".")[0]
        if sp not in all_sp_net and sp not in _vars: sp = col
        col_to_sp[col] = sp
        col_to_nH[col] = float(n_H_list[idx]) if idx < len(n_H_list) else 1.0

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "chi2_rmse":       _chi2_rmse,
        "ssr":             ssr,
        "n_points":        n_total,
        "n_params":        len(fit_keys),
        "param_values":    fitted_logKs,
        "param_errors":    param_errors_mixed,
        "param_cov":       _cov_mat,
        "param_cov_names": _cov_names_mixed,
        "sp_concs":        sp_concs,
        "col_to_sp":       col_to_sp,
        "col_to_nH":       col_to_nH,
        "pure_shifts":     pure_shifts,        # dd values (= δ − δ_obs(V=0))
        "pure_shifts_anchored": bool(_nmr_noref and _read_species),
        "nmr_noref":             bool(_nmr_noref),
        "delta_vecs_all":  delta_vecs_all,
        "delta_bound_all": delta_bound_all,
        "delta_free":      delta_free,
        "x_free_val":      x_free_val,
        "col_to_target":   col_to_target,
        "ref_corrections": ref_corrections,
        "fit_mode":        "mixed",
        "fitted_concs":    _fitted_concs_nmr,
        "fitted_titrants": _fitted_titrants_nmr,
        "n_iter":          getattr(result, "nit", 0),
        "timed_out":       timed_out,
        # Per-component stats (integration in mM, shift in ppm Δδ)
        "r2_integ":        r2_integ,
        "rmse_integ":      rmse_integ,
        "n_integ_pts":     n_integ_pts,
        "r2_shift":        r2_shift,
        "rmse_shift":      rmse_shift,
        "n_shift_pts":     n_shift_pts,
        # v2 diagnostics — combined flat residuals (integration block first, shift block second)
        "y_obs":           _mixed_yobs,
        "y_calc":          _mixed_ycalc,
        "residuals":       _mixed_res,
        # v2 bootstrap — per-source arrays (split for within-block resampling)
        "per_col_y_obs_integ":  per_col_y_obs_integ,
        "per_col_y_calc_integ": per_col_y_calc_integ,
        "per_col_x_integ":      per_col_x_integ,
        "per_col_y_obs_shift":  per_col_y_obs_shift,
        "per_col_y_calc_shift": per_col_y_calc_shift,
        "per_col_x_shift":      per_col_x_shift,
        # v2 unified per_col_* — what the residuals-vs-predictor scatter
        # consumes via collect_residuals_from_stats.  Same tagging
        # scheme as fit_kinetics_nmr_mixed so the legend distinguishes
        # integration from shift columns.
        "per_col_x":      {**{f"{c} (int)":   v for c, v in per_col_x_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_x_shift.items()}},
        "per_col_y_obs":  {**{f"{c} (int)":   v for c, v in per_col_y_obs_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_y_obs_shift.items()}},
        "per_col_y_calc": {**{f"{c} (int)":   v for c, v in per_col_y_calc_integ.items()},
                            **{f"{c} (shift)": v for c, v in per_col_y_calc_shift.items()}},
    }
    _r2   = stats.get("r_squared", 0.0)
    _conv = result.success or ssr < 1e-6 or (timed_out and _r2 >= 0.99)
    if timed_out and _r2 >= 0.99:
        stats["timed_out"] = False
    _msg = "Mixed NMR fit complete"
    return _conv, fitted_logKs, stats, _msg


# ─────────────────────────────────────────────
