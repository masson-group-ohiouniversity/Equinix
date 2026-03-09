"""equinix_kinetics_spectra.py"""
import time
import numpy as np
from scipy.optimize import minimize
from equinix_kinetics import compute_kinetics_curve, _collect_all_kinetic_species
from equinix_fit_spectra import _optimal_spectral_range
from equinix_fit_nmr import _get_species_for_target, _hessian_errors, _resolve_c
from equinix_kinetics_nmr import _kinetics_nmr_integration_backCalc
from equinix_curve import convert_exp_x
from equinix_parser import constraints_penalty

__all__ = ['fit_kinetics_spectra']


def fit_kinetics_spectra(parsed: dict, logk_dict: dict, spectra_data: dict,
                         fit_keys: list, t_max: float, n_pts_sim: int,
                         wl_min: float, wl_max: float,
                         tolerance: float, maxiter: int,
                         timeout_s: float = 30.0, auto_range: bool = False,
                         use_lbfgsb: bool = True, use_neldermead: bool = True,
                         constraints=None):
    """
    Fit rate constants to UV-Vis kinetics spectra using Beer-Lambert law.

    A(t, λ) = Σⱼ εⱼ(λ) · [Sⱼ](t)

    C(t) from ODE integration; E solved analytically by lstsq; k optimised by Nelder-Mead.
    """
    from scipy.optimize import minimize
    import time

    wavelengths = spectra_data["wavelengths"]
    t_exp       = spectra_data["x_vals"]
    A_full      = spectra_data["A"]

    spectra_cfg = parsed.get("spectra") or {}
    transparent = set(spectra_cfg.get("transparent", []))
    all_kin_sp  = _collect_all_kinetic_species(parsed)
    absorbers   = [sp for sp in all_kin_sp if sp not in transparent]

    if not absorbers:
        return False, {}, {}, "All species are transparent — nothing to fit"

    wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    A_fit   = A_full[:, wl_mask]
    if A_fit.shape[1] == 0:
        return False, {}, {}, "No wavelengths in selected range"

    n_pts  = len(t_exp)

    def _simulate(logk_trial):
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        try:
            return compute_kinetics_curve(parsed, lk, t_max, n_pts_sim)
        except Exception:
            return None

    def _build_C(curve):
        t_sim = curve["t"]
        C = np.zeros((n_pts, len(absorbers)))
        for j, sp in enumerate(absorbers):
            c_sim = _resolve_c(curve, sp, parsed, t_sim)
            C[:, j] = np.interp(t_exp, t_sim, c_sim)
        return np.maximum(C, 0.0)

    def objective(logk_trial):
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        cp = constraints_penalty(constraints or [], lk)
        if cp > 0:
            return cp
        curve = _simulate(logk_trial)
        if curve is None:
            return 1e12
        C = _build_C(curve)
        E, _, _, _ = np.linalg.lstsq(C, A_fit, rcond=None)
        return float(np.sum((A_fit - C @ E) ** 2))

    def data_objective(logk_trial):
        """Data-only SSR (no constraint penalty) — used for Hessian / error estimation."""
        curve = _simulate(logk_trial)
        if curve is None:
            return 1e12
        C = _build_C(curve)
        E, _, _, _ = np.linalg.lstsq(C, A_fit, rcond=None)
        return float(np.sum((A_fit - C @ E) ** 2))

    class _Timeout(Exception):
        pass

    x0  = np.array([logk_dict[k] for k in fit_keys])
    n_p = len(fit_keys)
    best_tracker = {"x": x0.copy(), "f": np.inf,
                    "start": time.time(), "timed_out": False}

    # ── Stage 1: L-BFGS-B (timeout-free, warm-starts Nelder-Mead) ────────
    # log k is unbounded in principle; use loose bounds to keep solver stable
    _k_lo = np.full(n_p, -6.0)   # log10(k) >= -6  →  k >= 1e-6
    _k_hi = np.full(n_p,  9.0)   # log10(k) <=  9  →  k <= 1e9
    bounds = list(zip(_k_lo.tolist(), _k_hi.tolist()))

    def objective_safe(logk_trial):
        penalty = sum(1e6*(v-lo)**2 for v,lo in zip(logk_trial,_k_lo) if v < lo) + \
                  sum(1e6*(v-hi)**2 for v,hi in zip(logk_trial,_k_hi) if v > hi)
        lk = logk_dict.copy()
        for i, k in enumerate(fit_keys):
            lk[k] = logk_trial[i]
        penalty += constraints_penalty(constraints or [], lk)
        if penalty > 0:
            return float(penalty)
        return objective(logk_trial)

    if use_lbfgsb:
        obj_start = objective_safe(x0)
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                res_lbfgs = minimize(objective_safe, x0, method="L-BFGS-B",
                                     bounds=bounds,
                                     options={"maxiter": maxiter,
                                              "ftol": tolerance,
                                              "gtol": tolerance * 1e-3})
            if res_lbfgs.fun < obj_start * 0.999:
                x0 = res_lbfgs.x
                best_tracker["x"] = x0.copy()
                best_tracker["f"] = res_lbfgs.fun
        except Exception:
            pass   # fall through to Nelder-Mead

    def _obj_timed(logk_trial):
        f = objective(logk_trial)
        if f < best_tracker["f"]:
            best_tracker["f"] = f
            best_tracker["x"] = logk_trial.copy()
        if time.time() - best_tracker["start"] > timeout_s:
            best_tracker["timed_out"] = True
            raise _Timeout()
        return f

    init_simplex = np.vstack([x0] + [x0 + np.eye(n_p)[i] * 1.5 for i in range(n_p)])
    if use_neldermead:
        try:
            result = minimize(_obj_timed, x0, method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": tolerance,
                                       "fatol": tolerance * 1e-4, "adaptive": True,
                                       "initial_simplex": init_simplex})
        except _Timeout:
            class _MockResult:
                x       = best_tracker["x"]
                success = False
                fun     = best_tracker["f"]
                nit     = 0
            result = _MockResult()
    else:
        # L-BFGS-B only — use its result directly
        class _MockResult:
            x       = best_tracker["x"]
            success = best_tracker["f"] < np.inf
            fun     = best_tracker["f"]
            nit     = 0
        result = _MockResult()

    # ── Auto-range pass 2 ────────────────────────────────────────────────
    if auto_range and len(absorbers) > 1:
        # Compute E from pass-1 result, find optimal wavelength window
        _curve1 = _simulate(result.x)
        if _curve1 is not None:
            _C1 = _build_C(_curve1)
            _E1, _, _, _ = np.linalg.lstsq(_C1, A_fit, rcond=None)
            wl_fit_now = wavelengths[wl_mask]
            wl_min, wl_max = _optimal_spectral_range(wl_fit_now, _E1, min_width_nm=50.0)
            wl_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            A_fit   = A_full[:, wl_mask]
            if A_fit.shape[1] > 0:
                best_tracker["x"] = result.x.copy()
                best_tracker["f"] = np.inf
                best_tracker["start"] = time.time()
                try:
                    result = minimize(_obj_timed, result.x, method="Nelder-Mead",
                                      options={"maxiter": maxiter, "xatol": tolerance,
                                               "fatol": tolerance * 1e-4, "adaptive": True})
                except _Timeout:
                    class _MockResult2:
                        x       = best_tracker["x"]
                        success = False
                        fun     = best_tracker["f"]
                        nit     = 0
                    result = _MockResult2()

    timed_out    = best_tracker["timed_out"]
    fitted_logks = {fit_keys[i]: result.x[i] for i in range(n_p)}

    # ── Final statistics ───────────────────────────────────────────────────
    curve_final = _simulate(result.x)
    if curve_final is None:
        return False, fitted_logks, {}, "ODE failed at fitted parameters"

    C_final = _build_C(curve_final)
    wl_fit  = wavelengths[wl_mask]
    E_final, _, _, _ = np.linalg.lstsq(C_final, A_fit, rcond=None)
    A_calc  = C_final @ E_final

    C_back, _, _, _ = np.linalg.lstsq(E_final.T, A_fit.T, rcond=None)
    C_back = np.clip(C_back.T, 0.0, None)

    residuals = (A_fit - A_calc).ravel()
    ssr  = float(np.sum(residuals ** 2))
    sst  = float(np.sum((A_fit - A_fit.mean()) ** 2))
    r2   = 1.0 - ssr / max(sst, 1e-30)
    rmse = float(np.sqrt(ssr / max(len(residuals), 1)))

    _c_res    = (C_back - C_final).ravel()
    _c_sst    = float(np.sum((C_back - C_back.mean()) ** 2))
    r2_conc   = float(1.0 - np.sum(_c_res ** 2) / max(_c_sst, 1e-30))
    rmse_conc = float(np.sqrt(np.sum(_c_res ** 2) / max(len(_c_res), 1)))

    _err_idx     = _hessian_errors(data_objective, result.x, ssr, len(residuals), n_p)
    param_errors = {fit_keys[i]: _err_idx[i] for i in range(n_p) if i in _err_idx}

    _r2_ok = r2 >= 0.99
    if timed_out and _r2_ok:
        timed_out = False
    _conv = getattr(result, "success", False) or ssr < 1e-6 or (not timed_out and _r2_ok)

    stats = {
        "r_squared":       r2,
        "rmse":            rmse,
        "ssr":             ssr,
        "n_points":        len(residuals),
        "n_params":        n_p,
        "param_values":    fitted_logks,
        "param_errors":    param_errors,
        "fit_mode":        "kinetics_spectra",
        "n_iter":          getattr(result, "nit", 0),
        "timed_out":       timed_out,
        "r2_conc":         r2_conc,
        "rmse_conc":       rmse_conc,
        "absorbers":       absorbers,
        "x_exp":           t_exp,
        "C_back":          C_back,
        "E_final":         E_final,
        "wavelengths_fit": wl_fit,
        "opt_wl_min":      wl_min,
        "opt_wl_max":      wl_max,
        "auto_range":      auto_range,
        "sp_concs": {}, "col_to_sp": {}, "col_to_nH": {},
        "pure_shifts": {}, "delta_vecs_all": {}, "delta_bound_all": {},
        "delta_free": {}, "x_free_val": {}, "col_to_target": {}, "ref_corrections": {},
    }
    return _conv, fitted_logks, stats, "Kinetics UV-Vis spectra fit complete"

