# -*- coding: utf-8 -*-
"""equilibrist_shift_constraints.py

Constrained intrinsic-shift solver for NMR fast-exchange fitting.

The Variable Projection (VP) step in fit_nmr / kinetics_nmr solves, per shift
column, X · d ≈ y where d is the vector of intrinsic species shifts (relative
to the reference species).  When many species feed into a single observed
shift, the linear subproblem has too much slack and absorbs speciation errors
that should constrain the K's.

This module lets the user tie or order intrinsic shifts of selected species
via the $constraints section:

    shift: G = GM = GM2          (equality group — same intrinsic shift)
    shift: GH > GHM > GHM2       (strict-decrease chain)
    shift: GH < GHM < GHM2       (strict-increase chain — equivalent to the reverse)

Public API:
    solve_shifts(X, y, non_free_species, ref_species, shift_constraints)
        -> (dd, calc, ssr)
        dd is shape (X.shape[1],) in the original column order; species
        merged into the same equality class share the same δ; species in the
        same class as the reference are forced to δ = 0.

Constraint dicts (as emitted by the parser):
    {"type": "shift_eq_group", "species": ["A", "B", "C", ...]}
    {"type": "shift_order",    "species": ["A", "B", "C", ...]}   # δ_A > δ_B > δ_C ...
"""

import numpy as np

try:
    from scipy.optimize import lsq_linear
    _HAS_LSQ_LINEAR = True
except Exception:  # pragma: no cover — scipy is a hard dep but be defensive
    _HAS_LSQ_LINEAR = False


# ─────────────────────────────────────────────────────────────────────────────
# Equality classes (union–find)
# ─────────────────────────────────────────────────────────────────────────────
def _build_eq_classes(non_free_species, ref_species, eq_groups):
    """Union species into equality classes.

    Returns:
        sp_to_cls   : {species_name: class_id}
        cls_members : [list_of_species_per_class_id]
    """
    universe = list(non_free_species) + [ref_species]
    parent = {s: s for s in universe}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for grp in eq_groups:
        members = [s for s in grp if s in parent]
        for s in members[1:]:
            union(members[0], s)

    classes = {}
    for s in universe:
        classes.setdefault(find(s), []).append(s)
    cls_id      = {root: i for i, root in enumerate(classes)}
    sp_to_cls   = {s: cls_id[find(s)] for s in universe}
    cls_members = [classes[root] for root in classes]
    return sp_to_cls, cls_members


# ─────────────────────────────────────────────────────────────────────────────
# Inequality DAG
# ─────────────────────────────────────────────────────────────────────────────
def _build_ineq_dag(cls_members, sp_to_cls, ineq_chains):
    """
    Convert pairwise inequalities (A > B) into a class-level DAG.

    Each chain [A, B, C, ...] generates edges A→B, B→C (parent → child, parent
    has the strictly larger δ).

    Validation:
      • Two species in the same equality class but adjacent in an inequality
        chain → contradiction → ValueError.
      • A class with two distinct inequality parents → unsupported by the
        substitution scheme → ValueError with a fix-it hint.
      • Cycles → ValueError.

    Returns:
        parent_of : {child_cls: parent_cls}  (single parent per child)
    """
    edges = []
    for chain in ineq_chains:
        cls_chain = []
        prev_sp_for_msg = []
        for s in chain:
            if s in sp_to_cls:
                cls_chain.append(sp_to_cls[s])
                prev_sp_for_msg.append(s)
        for i in range(len(cls_chain) - 1):
            a, b = cls_chain[i], cls_chain[i + 1]
            if a == b:
                raise ValueError(
                    f"shift-constraint contradiction: '{prev_sp_for_msg[i]}' and "
                    f"'{prev_sp_for_msg[i+1]}' are in the same equality group "
                    f"but appear as strictly different (>) on the same line."
                )
            edges.append((a, b, prev_sp_for_msg[i], prev_sp_for_msg[i + 1]))

    parents_of_child = {}
    children_of_parent = {}
    for p, c, sp_p, sp_c in edges:
        parents_of_child.setdefault(c, []).append((p, sp_p, sp_c))
        children_of_parent.setdefault(p, set()).add(c)

    # Reject classes with >1 distinct parent
    parent_of = {}
    for c, plist in parents_of_child.items():
        distinct = list({p for (p, _, _) in plist})
        if len(distinct) > 1:
            ex_names = [sp_p for (_, sp_p, _) in plist]
            child_name = plist[0][2]
            raise ValueError(
                f"shift-constraint: species '{child_name}' has multiple "
                f"inequality parents ({sorted(set(ex_names))}). "
                f"This isn't supported — combine the parents into one equality "
                f"group ('shift: {' = '.join(sorted(set(ex_names)))}'), or drop "
                f"one of the inequalities."
            )
        parent_of[c] = distinct[0]

    # Cycle detection
    color = {n: 0 for n in range(len(cls_members))}  # 0=white, 1=gray, 2=black

    def dfs(n, stack):
        if color[n] == 1:
            cycle_names = [cls_members[k][0] for k in stack]
            raise ValueError(
                f"shift-constraint: cyclic inequality detected through species "
                f"{cycle_names}."
            )
        if color[n] == 2:
            return
        color[n] = 1
        for nxt in children_of_parent.get(n, ()):
            dfs(nxt, stack + [nxt])
        color[n] = 2

    for n in range(len(cls_members)):
        if color[n] == 0:
            dfs(n, [n])

    return parent_of


# ─────────────────────────────────────────────────────────────────────────────
# Reparameterisation
# ─────────────────────────────────────────────────────────────────────────────
def _build_theta_transform(non_ref_classes, cls_col_idx, ref_cls, parent_of):
    """
    Build a transformation T (Δ_class = T · θ) and per-θ bounds.

    Each θ-component is either:
      • a "base" free parameter (unbounded), or
      • a "slack" (≥ 0) representing α in Δ_child = Δ_parent − α.

    Substitution rules per non-ref class k:
      parent = None and (k is parent of ref_cls):  Δ_k base, lb = 0  (k > ref)
      parent = None and (otherwise):               Δ_k base, unbounded
      parent = ref_cls:                            Δ_k = -α   (α ≥ 0 ⇒ Δ_k ≤ 0)
      parent = non-ref class p:                    Δ_k = Δ_p − α  (α ≥ 0)
    """
    # Topo order of non-ref classes (parents before children)
    order = []
    seen = set()

    def visit(k):
        if k in seen:
            return
        if k == ref_cls:
            seen.add(k)
            return
        p = parent_of.get(k)
        if p is not None and p != ref_cls:
            visit(p)
        seen.add(k)
        order.append(k)

    for k in non_ref_classes:
        visit(k)

    parent_of_ref = parent_of.get(ref_cls)  # may be None
    delta_combo = {}    # cls_id → list of (theta_idx, sign)
    theta_bounds = []   # list of (lb, ub)

    for k in order:
        p = parent_of.get(k)
        if p is None:
            lb = 0.0 if parent_of_ref == k else -np.inf
            theta_bounds.append((lb, np.inf))
            delta_combo[k] = [(len(theta_bounds) - 1, +1.0)]
        elif p == ref_cls:
            theta_bounds.append((0.0, np.inf))
            delta_combo[k] = [(len(theta_bounds) - 1, -1.0)]
        else:
            theta_bounds.append((0.0, np.inf))
            slack_idx = len(theta_bounds) - 1
            delta_combo[k] = list(delta_combo[p]) + [(slack_idx, -1.0)]

    n_theta = len(theta_bounds)
    n_cls   = len(non_ref_classes)
    T = np.zeros((n_cls, n_theta))
    for k in non_ref_classes:
        j = cls_col_idx[k]
        for t_idx, sign in delta_combo[k]:
            T[j, t_idx] += sign

    lbs = np.array([b[0] for b in theta_bounds])
    ubs = np.array([b[1] for b in theta_bounds])
    return T, lbs, ubs, delta_combo


# ─────────────────────────────────────────────────────────────────────────────
# Public solver
# ─────────────────────────────────────────────────────────────────────────────
def get_per_column_bounds(shift_constraints):
    """
    Return the per-column bounds list, or None if absent.

    A 'shift: a, b, free, c, d' line in $constraints emits:
        {"type": "shift_per_column_bounds", "values": [a, b, None, c, d]}

    If multiple such entries exist, the *last* one wins.
    """
    found = None
    for c in shift_constraints or ():
        if c.get("type") == "shift_per_column_bounds":
            found = c.get("values")
    return found


def solve_shifts(X, y, species, ref_species, shift_constraints,
                 column_bound=None, pinned_dd=None, noref=False):
    """
    Solve X·dd ≈ y for intrinsic species shifts under user constraints.

    Parameters
    ----------
    X : ndarray, shape (n_pts, n_species)
        Full design matrix: ``X[t, i] = f_i(t) − f_i(V=0)`` where ``f_i`` is
        the mole fraction of species i within the target.  Includes the
        reference species' column.
    y : ndarray, shape (n_pts,)
        ``δ_obs(t) − δ_obs(V=0)``.
    species : list[str]
        Species names matching X's columns (length n_species, includes ``ref_species``).
    ref_species : str
        Name of the math-reference species (must be in ``species``).
    shift_constraints : list[dict] or None
        Constraints from $constraints (eq groups, ineq chains, per-column bounds).
    column_bound : float or None
        Per-column bound on dd values.
    pinned_dd : dict[str, float] or None
        User-supplied pins from ``$nmr read:`` species (mapping species name to
        ``sheet2_value − δ_obs(V=0)``).
    noref : bool, default False
        If False (legacy / default): the reference species' dd is automatically
        pinned to 0 in addition to any user pins.  Output should be interpreted
        as "Δδ relative to ref".  This is the behavior expected by tutorials
        and by scripts that don't include ``noref`` in $nmr.

        If True: NO automatic ref pin.  The reference species participates in
        the LS just like every other species.  At least one user pin (from
        ``$nmr read:``) is required to anchor the absolute shift scale —
        without one, the LS is rank-1 deficient (uniform-shift symmetry) and
        a ValueError is raised.  Output can be interpreted as "absolute δ"
        once the ``δ_obs(V=0)`` baseline is added back.

    Returns
    -------
    dd : ndarray, shape (n_species,)
        ``dd[i] = δ_i_absolute − δ_obs(V=0)`` for every species in input order.
    calc : ndarray, shape (n_pts,)
        ``X @ dd``.
    ssr : float
        Sum of squared residuals.

    Raises
    ------
    ValueError
        On contradictory constraints (cycles, multi-parents, conflicting pins,
        pins outside bounds, ``noref=True`` with no user pin, …).
    """
    n_species = X.shape[1]
    if n_species == 0:
        return np.array([]), np.zeros_like(y), float(np.sum(y ** 2))
    if len(species) != n_species:
        raise ValueError(
            f"solve_shifts: len(species)={len(species)} != X.shape[1]={n_species}"
        )
    if ref_species not in species:
        raise ValueError(
            f"solve_shifts: ref_species '{ref_species}' not in species list."
        )

    # ── Filter constraints to species relevant for this column ───────────────
    relevant = set(species)
    eq_groups, ineq_chains = [], []
    for c in shift_constraints or ():
        t = c.get("type")
        if t == "shift_eq_group":
            members = [s for s in c.get("species", ()) if s in relevant]
            if len(members) >= 2:
                eq_groups.append(members)
        elif t == "shift_order":
            members = [s for s in c.get("species", ()) if s in relevant]
            if len(members) >= 2:
                ineq_chains.append(members)

    # ── Validate composition ────────────────────────────────────────────────
    if column_bound is not None and ineq_chains:
        raise ValueError(
            "Per-column shift bound (e.g. 'shift: -0.8, -0.5, ...') cannot be "
            "combined with strict-inequality constraints ('>' or '<') for the "
            "same column. Remove one of them."
        )

    # ── Build the effective pin set ─────────────────────────────────────────
    # Default (noref=False, legacy):     auto-pin ref to dd=0, then layer
    #                                    user pins on top.
    # noref=True:                        no auto-pin; require at least one
    #                                    user pin to anchor the scale.
    pin_local = {}
    if pinned_dd:
        for sp, val in pinned_dd.items():
            if sp in relevant:
                try:
                    pin_local[sp] = float(val)
                except (TypeError, ValueError):
                    pass  # drop unparseable

    if noref:
        # No automatic pin to the math reference.  If the user provided pins
        # (via $nmr 'read:'), they're the only constraints.  If the user
        # provided NO pins, the LS is rank-1 deficient (uniform-shift
        # symmetry), but lstsq returns a well-defined minimum-norm solution
        # — the K-values and relative shifts are fully determined regardless.
        # Callers that want a "relative to ref" display should subtract
        # dd[ref] from the result (cosmetic; doesn't affect K's).
        pass
    else:
        # Auto-pin ref (legacy behavior).  If the user also pinned ref via
        # read: with a non-zero dd, that's a contradiction we catch below.
        if ref_species in pin_local:
            if abs(pin_local[ref_species]) > 1e-9:
                raise ValueError(
                    f"$nmr read: pin for the reference species '{ref_species}' "
                    f"requires dd = {pin_local[ref_species]:+.4f}, but in "
                    f"default (no-noref) mode the reference is auto-pinned to "
                    f"dd = 0.  This is a contradiction.  To use a non-zero ref "
                    f"shift, add the 'noref' line in $nmr."
                )
            # Else: user pinned ref to 0, which agrees with auto-pin — fine.
        pin_local[ref_species] = 0.0

    user_provided_pin = bool(pinned_dd)  # used only for the bound-vs-pin check

    # Pins on species inside inequality chains are not supported.
    if pin_local and ineq_chains:
        bad = set(pin_local) & {sp for chain in ineq_chains for sp in chain}
        # Only flag user-supplied pins (auto-pin to ref is OK if ref is in chain
        # because ineq chain ordering treats ref specially anyway — but to be
        # safe, accept any pin here and let the eq-class merge logic handle it).
        bad &= set(pinned_dd or {})
        if bad:
            raise ValueError(
                f"Pinned species {sorted(bad)} appear in inequality "
                f"constraints ('>' or '<'). Pinning + strict inequality on "
                f"the same species is not supported — remove one."
            )

    # ── Pin value vs per-column bound check ─────────────────────────────────
    # Per-column bound semantics: |dd| <= |column_bound|, i.e. dd ∈ [-B, +B]
    # for B = abs(column_bound).  (Legacy scripts wrote signed bounds; the
    # parser now stores |B| so the sign here is informational only.)
    if user_provided_pin and column_bound is not None:
        B = abs(float(column_bound))
        lo, hi = -B, B
        for sp in (pinned_dd or {}):
            if sp not in pin_local: continue
            val = pin_local[sp]
            if val < lo - 1e-9 or val > hi + 1e-9:
                raise ValueError(
                    f"Pinned dd for '{sp}' ({val:+.4f}) lies outside the "
                    f"per-column bound [{lo:+.2f}, {hi:+.2f}].  Either widen "
                    f"the bound or remove the read: pin for this column."
                )

    # ── Equality classes ────────────────────────────────────────────────────
    sp_to_cls, cls_members = _build_eq_classes_full(species, eq_groups)
    n_cls = len(cls_members)

    # ── Propagate pins through equality classes & validate ──────────────────
    cls_pin = {}
    for sp, val in pin_local.items():
        cls_id = sp_to_cls[sp]
        if cls_id in cls_pin:
            if abs(cls_pin[cls_id] - val) > 1e-6:
                conflicting = [s for s in cls_members[cls_id] if s in pin_local]
                raise ValueError(
                    f"Conflicting pinned dd values within equality class "
                    f"{cls_members[cls_id]}: species {conflicting} pinned "
                    f"to inconsistent values "
                    f"({[pin_local[s] for s in conflicting]}). Either "
                    f"reconcile the values or drop one of the equality lines."
                )
        else:
            cls_pin[cls_id] = val

    free_classes = [k for k in range(n_cls) if k not in cls_pin]
    cls_col_idx  = {k: j for j, k in enumerate(free_classes)}

    # ── Merge X columns by class; subtract pinned contributions from y ──────
    sp_idx = {s: i for i, s in enumerate(species)}
    y_adj = y.astype(float).copy()
    for k, dpin in cls_pin.items():
        col_vec = np.zeros(X.shape[0])
        for s in cls_members[k]:
            col_vec += X[:, sp_idx[s]]
        y_adj -= col_vec * dpin

    merged_X = np.zeros((X.shape[0], len(free_classes)))
    for k in free_classes:
        col = cls_col_idx[k]
        for s in cls_members[k]:
            merged_X[:, col] += X[:, sp_idx[s]]

    # If every class is pinned, all dd are determined — no LS needed
    if merged_X.shape[1] == 0:
        dd = np.zeros(n_species)
        for k in range(n_cls):
            d_k = float(cls_pin.get(k, 0.0))
            for s in cls_members[k]:
                dd[sp_idx[s]] = d_k
        calc = X @ dd
        return dd, calc, float(np.sum((y - calc) ** 2))

    # ── Inequality DAG → reparameterisation (over free classes only) ───────
    free_ineq = []
    for chain in ineq_chains:
        cls_chain = [sp_to_cls[s] for s in chain if s in sp_to_cls]
        if any(c in cls_pin for c in cls_chain):
            continue
        free_ineq.append(chain)

    pinned_cls_root = next(iter(cls_pin)) if cls_pin else None
    parent_of = _build_ineq_dag(cls_members, sp_to_cls, free_ineq)
    T, lbs, ubs, delta_combo = _build_theta_transform(
        free_classes, cls_col_idx, pinned_cls_root, parent_of
    )

    # ── Apply per-column bound (symmetric: |dd| <= B) ──────────────────────
    if column_bound is not None:
        B = abs(float(column_bound))
        cls_lb_b, cls_ub_b = -B, B
        for k in free_classes:
            for t_idx, sign in delta_combo[k]:
                # delta_combo[k] expresses class k's dd as a signed sum of
                # theta variables.  When sign>0, theta_t inherits the class
                # bound directly; when sign<0, the bound flips.  Symmetric
                # bounds [-B, +B] are unaffected by sign flips, so this
                # collapses to the same [-B, +B] range either way.
                lbs[t_idx] = max(lbs[t_idx], -B)
                ubs[t_idx] = min(ubs[t_idx],  B)

    # ── Solve (against the pin-adjusted y) ──────────────────────────────────
    A = merged_X @ T
    is_unbounded = np.all(np.isneginf(lbs)) and np.all(np.isposinf(ubs))
    if is_unbounded or not _HAS_LSQ_LINEAR:
        theta = np.linalg.lstsq(A, y_adj, rcond=None)[0]
        theta = np.minimum(np.maximum(theta, lbs), ubs)
    else:
        try:
            res = lsq_linear(A, y_adj, bounds=(lbs, ubs), method="bvls",
                             max_iter=200)
            theta = res.x
        except Exception:
            res = lsq_linear(A, y_adj, bounds=(lbs, ubs), method="trf",
                             max_iter=200)
            theta = res.x

    # ── Reconstruct per-class δ, then per-species dd ───────────────────────
    delta_cls_free = T @ theta
    dd = np.zeros(n_species)
    for k in range(n_cls):
        if k in cls_pin:
            d_k = float(cls_pin[k])
        else:
            d_k = float(delta_cls_free[cls_col_idx[k]])
        for s in cls_members[k]:
            dd[sp_idx[s]] = d_k

    calc = X @ dd
    return dd, calc, float(np.sum((y - calc) ** 2))


def _build_eq_classes_full(species, eq_groups):
    """Like _build_eq_classes but takes a flat species list (no separate ref).

    Returns:
        sp_to_cls   : {species_name: class_id}
        cls_members : [list_of_species_per_class_id]
    """
    parent = {s: s for s in species}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for grp in eq_groups:
        members = [s for s in grp if s in parent]
        for s in members[1:]:
            union(members[0], s)

    classes = {}
    for s in species:
        classes.setdefault(find(s), []).append(s)
    cls_id      = {root: i for i, root in enumerate(classes)}
    sp_to_cls   = {s: cls_id[find(s)] for s in species}
    cls_members = [classes[root] for root in classes]
    return sp_to_cls, cls_members
