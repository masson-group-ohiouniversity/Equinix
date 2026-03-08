"""equinix_fit_conc.py"""
import numpy as np
from scipy.optimize import minimize
from equinix_curve import compute_single_point, find_equiv_for_x, convert_exp_x

__all__ = ['fit_parameters']


def fit_parameters(parsed, network, exp_data, params, logK_vals, fit_keys, x_expr, tolerance=1e-6, maxiter=100_000, use_lbfgsb=True, use_neldermead=True):
    """
    Fit selected log K parameters to experimental data with robust error handling.
    
    Returns (success: bool, fitted_logKs: dict, stats: dict, message: str)
    """
    try:
        if not exp_data or not fit_keys:
            return False, {}, {}, "No experimental data or parameters to fit"
        
        # Starting values for fitting (only the parameters to fit)
        fit_param_names = list(fit_keys)
        x0 = np.array([logK_vals[k] for k in fit_param_names])
        
        # Pre-compute experimental points and cache x-to-equiv conversions for speed
        all_exp_x = []
        all_exp_y = []
        exp_points = []  # [(x_val, y_val, column_name), ...]
        x_to_equiv_cache = {}  # Simple cache for speed
        
        for exp_col, col_data in exp_data.items():
            if exp_col.startswith("_"): continue
            try:
                v_add_mL = col_data["v_add_mL"]
                exp_y = col_data["y"]
                
                # Convert volumes to x-axis values
                exp_x = convert_exp_x(v_add_mL, x_expr, parsed, params, network)
                
                for i in range(len(exp_x)):
                    if np.isfinite(exp_x[i]) and np.isfinite(exp_y[i]):
                        x_val = exp_x[i]
                        exp_points.append((x_val, exp_y[i], exp_col))
                        all_exp_x.append(x_val)
                        all_exp_y.append(exp_y[i])
                        # Pre-cache equiv conversion for speed
                        if x_val not in x_to_equiv_cache:
                            x_to_equiv_cache[x_val] = find_equiv_for_x(x_val, parsed, params)
                        
            except Exception as e:
                continue  # Skip problematic columns
        
        if len(exp_points) < len(fit_param_names):
            return False, {}, {}, f"Too few valid data points ({len(exp_points)}) for {len(fit_param_names)} parameters"
        
        # Safe logK range: covers all chemistry, prevents exp overflow in solver.
        LOGK_MIN, LOGK_MAX = -15.0, 15.0

        def objective(fit_params):
            """Overflow-safe objective: suppress numpy warnings, clip logK, return
            finite large value on any failure so gradient-based optimizers stay stable."""
            try:
                current_logKs = logK_vals.copy()
                for i, pname in enumerate(fit_param_names):
                    current_logKs[pname] = float(np.clip(fit_params[i], LOGK_MIN, LOGK_MAX))
                residuals = []
                # Suppress overflow warnings — they're handled by the isfinite check below
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    for exp_x, exp_y, exp_col in exp_points:
                        try:
                            equiv = x_to_equiv_cache.get(exp_x, find_equiv_for_x(exp_x, parsed, params))
                            theo  = compute_single_point(equiv, parsed, network, current_logKs, params, exp_col)
                            residuals.append((exp_y - theo)**2 if np.isfinite(theo) else 1e6)
                        except Exception:
                            residuals.append(1e6)
                if not residuals:
                    return 1e9
                total = float(np.sum(residuals))
                return total if np.isfinite(total) else 1e9
            except Exception:
                return 1e9

        obj_start = objective(x0)
        if not np.isfinite(obj_start):
            return False, {}, {}, "Initial objective function evaluation failed"

        bounds = [(LOGK_MIN, LOGK_MAX) for _ in fit_param_names]

        # ── Stage 1: L-BFGS-B (if enabled) ─────────────────────────────────────
        # Fast, accurate, ideal for 1-8 parameters. np.errstate in objective keeps
        # overflow from producing NaN, so line search stays numerically stable.
        improved  = False
        converged = False
        best_x    = x0.copy()
        result    = None

        if use_lbfgsb:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': maxiter, 'ftol': tolerance, 'gtol': tolerance})
            obj_at_result = objective(result.x)
            improved  = np.isfinite(obj_at_result) and (obj_at_result < obj_start * 0.999)
            converged = result.success or improved
            if improved:
                best_x = result.x

        # ── Stage 2: Nelder-Mead (if enabled) ───────────────────────────────────
        # Derivative-free, structurally immune to gradient/line-search failures.
        # Runs as fallback when both enabled + L-BFGS-B failed, OR as sole
        # optimizer when L-BFGS-B is unchecked.
        if use_neldermead and not improved:
            result_nm = minimize(objective, x0, method='Nelder-Mead',
                                 options={'maxiter': maxiter,
                                          'xatol': max(tolerance ** 0.5, 1e-6),
                                          'fatol': max(tolerance, 1e-8),
                                          'adaptive': True})
            obj_nm = objective(result_nm.x)
            if np.isfinite(obj_nm) and obj_nm < obj_start * 0.999:
                best_x    = result_nm.x
                improved  = True
                converged = True
                result    = result_nm

        if result is None:
            return False, {}, {}, "No optimizer was enabled" 

        fitted_logKs = {}
        for i, param_name in enumerate(fit_param_names):
            fitted_logKs[param_name] = float(np.clip(best_x[i], LOGK_MIN, LOGK_MAX))
        
        # Calculate fit statistics with error handling
        try:
            final_logKs = logK_vals.copy()
            final_logKs.update(fitted_logKs)
            
            final_residuals = []
            theo_vals = []
            exp_vals = []
            
            for exp_x, exp_y, exp_col in exp_points:
                try:
                    equiv = x_to_equiv_cache.get(exp_x, find_equiv_for_x(exp_x, parsed, params))
                    theo_conc = compute_single_point(equiv, parsed, network, final_logKs, params, exp_col)
                    
                    if np.isfinite(theo_conc):
                        final_residuals.append((exp_y - theo_conc)**2)
                        theo_vals.append(theo_conc)
                        exp_vals.append(exp_y)
                except Exception:
                    continue
            
            if len(final_residuals) == 0:
                return False, {}, {}, "No valid residuals for statistics calculation"
            
            ssr = np.sum(final_residuals)
            rmse = np.sqrt(ssr / len(final_residuals))
            
            # Calculate R²
            if len(exp_vals) > 0:
                exp_mean = np.mean(exp_vals)
                sst = np.sum((np.array(exp_vals) - exp_mean)**2)
                r_squared = 1 - (ssr / max(sst, 1e-12)) if sst > 1e-12 else 0.0
            else:
                r_squared = 0.0
            
            stats = {
                "r_squared":   float(r_squared),
                "rmse":        float(rmse),
                "n_points":    len(exp_points),
                "n_params":    len(fit_param_names),
                "ssr":         float(ssr),
                "n_iter":      int(getattr(result, '_best_nit', result.nit)),
                "param_errors": {},   # filled below
                "param_values": {},   # fitted values for display
            }

            # ── Parameter uncertainties via numerical Jacobian ───────────
            # Build residual vector at solution (without penalty term)
            def raw_residuals_vec(fit_params):
                res = []
                current_logKs = logK_vals.copy()
                for i, pname in enumerate(fit_param_names):
                    current_logKs[pname] = fit_params[i]
                for exp_x, exp_y, exp_col in exp_points:
                    try:
                        equiv    = x_to_equiv_cache.get(exp_x, find_equiv_for_x(exp_x, parsed, params))
                        theo     = compute_single_point(equiv, parsed, network, current_logKs, params, exp_col)
                        res.append(exp_y - theo if np.isfinite(theo) else 0.0)
                    except Exception:
                        res.append(0.0)
                return np.array(res)

            try:
                n_pts_valid = len(exp_points)
                n_par       = len(fit_param_names)
                if n_pts_valid > n_par:
                    x_sol = best_x
                    eps   = 1e-4   # step in log K units
                    J     = np.zeros((n_pts_valid, n_par))
                    r0    = raw_residuals_vec(x_sol)
                    for k in range(n_par):
                        dx        = np.zeros(n_par)
                        dx[k]     = eps
                        J[:, k]   = (raw_residuals_vec(x_sol + dx) - r0) / eps

                    # σ² = SSR / (n - p),  cov ≈ σ² (JᵀJ)⁻¹
                    sigma2 = float(np.dot(r0, r0)) / (n_pts_valid - n_par)
                    JtJ    = J.T @ J
                    try:
                        cov     = np.linalg.inv(JtJ) * sigma2
                        param_errors = {
                            fit_param_names[k]: float(np.sqrt(max(cov[k, k], 0.0)))
                            for k in range(n_par)
                        }
                    except np.linalg.LinAlgError:
                        # singular JtJ — use diagonal pseudo-inverse
                        diag_inv = np.where(np.diag(JtJ) > 0, 1.0 / np.diag(JtJ), 0.0)
                        param_errors = {
                            fit_param_names[k]: float(np.sqrt(max(diag_inv[k] * sigma2, 0.0)))
                            for k in range(n_par)
                        }
                    stats["param_errors"] = param_errors
            except Exception:
                pass   # errors stay empty dict — not fatal

            for k, pname in enumerate(fit_param_names):
                stats["param_values"][pname] = float(best_x[k])
            
            msg = "Fit successful" if converged else f"Converged with warnings: {result.message}"
            return converged, fitted_logKs, stats, msg
            
        except Exception as e:
            return False, {}, {}, f"Statistics calculation failed: {str(e)}"
        
    except Exception as e:
        return False, {}, {}, f"Fitting error: {str(e)}"

