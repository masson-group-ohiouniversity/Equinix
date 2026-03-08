"""equinix_parser.py"""
import re
import math
from collections import Counter
import numpy as np

__all__ = ['parse_value_with_units', 'convert_concentration_to_mM', 'convert_volume_to_mL', 'parse_equilibrium_constant', 'parse_species_with_stoich', 'parse_script', 'check_script_syntax']


def parse_value_with_units(value_str):
    """
    Parse a value string that may contain units.
    
    Examples:
        "1.50"     → (1.50, None)
        "1.50 mM"  → (1.50, "mM") 
        "500 uL"   → (500, "uL")
        "1e-6 M"   → (1e-6, "M")
    
    Returns (value, unit) where unit is None if not specified
    """
    import re
    value_str = value_str.strip()
    
    # Match number (including scientific notation) optionally followed by units
    match = re.match(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(\w+)?$', value_str)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else None
        return value, unit
    else:
        # Fallback: try to parse as pure number
        try:
            return float(value_str), None
        except ValueError:
            raise ValueError(f"Could not parse value: '{value_str}'")

def convert_concentration_to_mM(value, unit):
    """Convert concentration to mM (internal standard)."""
    if unit is None or unit == "mM":
        return value  # Already in mM (default)
    elif unit == "M":
        return value * 1000  # M to mM
    elif unit == "uM":
        return value / 1000  # uM to mM
    else:
        raise ValueError(f"Unsupported concentration unit: {unit}")

def convert_volume_to_mL(value, unit):
    """Convert volume to mL (internal standard)."""
    if unit is None or unit == "mL":
        return value  # Already in mL (default)
    elif unit == "L":
        return value * 1000  # L to mL
    elif unit == "uL":
        return value / 1000  # uL to mL
    else:
        raise ValueError(f"Unsupported volume unit: {unit}")

def parse_equilibrium_constant(k_str):
    """
    Parse equilibrium constant that may be log K or K.
    
    Examples:
        "log K = 1.50"  → (1.50, True)   # Already log K
        "Log K = 1.50"  → (1.50, True)   # Case insensitive
        "K = 1e5"       → (5.0, False)   # Converted to log K
    
    Returns (log_k_value, was_already_log)
    """
    import re
    import math
    k_str = k_str.strip()
    
    # Check for "log K" pattern (case insensitive)
    log_match = re.match(r'^log\s+k\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$', k_str, re.IGNORECASE)
    if log_match:
        log_k = float(log_match.group(1))
        return log_k, True
    
    # Check for plain "K" pattern
    k_match = re.match(r'^k\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$', k_str, re.IGNORECASE)
    if k_match:
        k_value = float(k_match.group(1))
        if k_value <= 0:
            raise ValueError(f"Equilibrium constant must be positive: K = {k_value}")
        log_k = math.log10(k_value)
        return log_k, False
    
    raise ValueError(f"Could not parse equilibrium constant: '{k_str}'")

# ─────────────────────────────────────────────
# EXCEL EXPORT FUNCTIONALITY 
# ─────────────────────────────────────────────


def parse_species_with_stoich(species_str):
    """Parse species string with stoichiometric coefficient.
    
    REQUIRES SPACE between coefficient and species name to distinguish from
    numbers that are part of the species name.
    
    Examples:
        "A" → (1, "A")
        "GH2M" → (1, "GH2M")     ← Numbers in name, no coefficient  
        "2 A" → (2, "A")          ← Coefficient 2, species A
        "2 GH2M" → (2, "GH2M")    ← Coefficient 2, species GH2M
        "10 H2O" → (10, "H2O")    ← Coefficient 10, species H2O
        "CO2" → (1, "CO2")        ← Just a species name
    """
    import re
    species_str = species_str.strip()
    
    # Look for pattern: digits + SPACE + species name
    match = re.match(r'(\d+)\s+([A-Za-z]\w*)', species_str)
    if match:
        coeff_str, species = match.groups()
        coefficient = int(coeff_str)
        return coefficient, species
    else:
        # No space-separated coefficient found, entire string is species name
        return 1, species_str


def parse_script(text: str) -> dict:
    result = {
        "concentrations":   {},
        "volumes":          {},
        "titrant":          {},
        "titrant_is_solid": False,
        "titrant_ratios":   {},
        "equilibria":       [],      # pre-equilibrium reactions (= type)
        "kinetics":         [],      # kinetic reactions (> and <> types)
        "is_kinetics":      False,   # True if any > or <> found
        "variables":        {},
        "plot_xmax":        3.0,
        "plot_x_expr":      None,
        "plot_y":           [],
        "nmr":              None,   # None | {"mode": "shift"|"integration", "targets": [str]}
        "spectra":          None,   # None | {"transparent": [str]}
    }

    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#")[0].strip()
        if not line:
            continue

        if line.startswith("$"):
            section_raw = line[1:].lower().strip()
            if section_raw.startswith("titrant"):
                section = "titrant"
                result["titrant_is_solid"] = "solid" in section_raw
            elif section_raw == "reactions":
                section = "equilibria"   # map new name to internal key
            elif section_raw == "spectra":
                section = "spectra"
                if result["spectra"] is None:
                    result["spectra"] = {"transparent": []}  # init on section entry
            else:
                section = section_raw
            continue

        num_re = r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?"

        if section == "concentrations":
            m = re.match(r"(\w+)\s*=\s*(.+)", line)
            if m:
                name = m.group(1)
                value_str = m.group(2).strip()
                try:
                    value, unit = parse_value_with_units(value_str)
                    # Convert to mM (internal standard)
                    mM_value = convert_concentration_to_mM(value, unit)
                    result["concentrations"][name] = mM_value
                except ValueError as e:
                    raise ValueError(f"Error parsing concentration '{line}': {e}")

        elif section == "volumes":
            m = re.match(r"(\w+)\s*=\s*(.+)", line)
            if m:
                name = m.group(1)
                value_str = m.group(2).strip()
                try:
                    value, unit = parse_value_with_units(value_str)
                    # Convert to mL (internal standard)
                    mL_value = convert_volume_to_mL(value, unit)
                    result["volumes"][name] = mL_value
                except ValueError as e:
                    raise ValueError(f"Error parsing volume '{line}': {e}")

        elif section == "titrant":
            if result["titrant_is_solid"]:
                # Solid syntax: "Mt" or "Mt; 1.3"
                parts = line.split(";")
                tkey  = parts[0].strip()
                ratio = float(parts[1].strip()) if len(parts) > 1 else 1.0
                # Store sentinel 0.0 as stock conc (not used), record ratio separately
                result["titrant"][tkey]        = 0.0
                result["titrant_ratios"][tkey] = ratio
            else:
                # Liquid syntax: "Mt = 10.00"
                m = re.match(r"(\w+)\s*=\s*(.+)", line)
                if m:
                    name = m.group(1)
                    value_str = m.group(2).strip()
                    try:
                        value, unit = parse_value_with_units(value_str)
                        mM_value = convert_concentration_to_mM(value, unit)
                        result["titrant"][name] = mM_value
                    except ValueError as e:
                        raise ValueError(f"Error parsing titrant '{line}': {e}")

        elif section == "equilibria":
            if ";" not in line:
                raise ValueError(f"Missing ';' in equilibrium line: '{line}'")
            
            parts    = line.split(";")
            rxn_part = parts[0].strip()
            k_parts  = [p.strip() for p in parts[1:] if p.strip()]

            # ── Detect reaction type ─────────────────────────────────
            if "<>" in rxn_part:
                lhs, rhs_rxn = rxn_part.split("<>", 1)
                rxn_type = "reversible_kinetic"
                result["is_kinetics"] = True
            elif ">" in rxn_part:
                lhs, rhs_rxn = rxn_part.split(">", 1)
                rxn_type = "irreversible"
                result["is_kinetics"] = True
            elif "=" in rxn_part:
                lhs, rhs_rxn = rxn_part.split("=", 1)
                rxn_type = "equilibrium"
            else:
                raise ValueError(f"No reaction arrow (=, >, <>) in: '{line}'")

            # ── Parse species lists ──────────────────────────────────
            def _parse_side(side_str):
                side_str = side_str.strip()
                if "+" in side_str:
                    return [p.strip() for p in side_str.split(" + ") if p.strip()]
                return [side_str] if side_str else []

            reactants_str = _parse_side(lhs)
            products_str  = _parse_side(rhs_rxn)

            reactants = [parse_species_with_stoich(r) for r in reactants_str]
            products  = [parse_species_with_stoich(p) for p in products_str]

            # ── Parse constants ──────────────────────────────────────
            import math

            if rxn_type == "equilibrium":
                # Existing: "log K = 5.0" or "K = 1e3"
                if not k_parts:
                    raise ValueError(f"Missing K in: '{line}'")
                kp = k_parts[0]
                k_match = re.match(r"^(log\s+)?([A-Za-z0-9_,]+)\s*=\s*(.+)$", kp, re.IGNORECASE)
                if not k_match:
                    raise ValueError(f"Cannot parse K in: '{line}'")
                is_log   = k_match.group(1) is not None
                kname    = k_match.group(2)
                kval_str = k_match.group(3)
                kval     = float(kval_str)
                logK     = kval if is_log else math.log10(kval)

                # ── Optional soft bounds: "from X to Y", "from X", "to Y" ──
                _num_pat = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
                _bound_lo = None
                _bound_hi = None
                for _bp in k_parts[1:]:
                    _bp = _bp.strip()
                    _m_from_to = re.match(
                        rf"from\s+({_num_pat})\s+to\s+({_num_pat})", _bp, re.IGNORECASE)
                    _m_from    = re.match(rf"from\s+({_num_pat})$", _bp, re.IGNORECASE)
                    _m_to      = re.match(rf"to\s+({_num_pat})$", _bp, re.IGNORECASE)
                    if _m_from_to:
                        _lo = float(_m_from_to.group(1))
                        _hi = float(_m_from_to.group(2))
                        _bound_lo = _lo if is_log else math.log10(_lo)
                        _bound_hi = _hi if is_log else math.log10(_hi)
                    elif _m_from:
                        _lo = float(_m_from.group(1))
                        _bound_lo = _lo if is_log else math.log10(_lo)
                    elif _m_to:
                        _hi = float(_m_to.group(1))
                        _bound_hi = _hi if is_log else math.log10(_hi)

                result["equilibria"].append({
                    "type":      "equilibrium",
                    "reactants": reactants,
                    "products":  products,
                    "kname":     kname,
                    "logK":      logK,
                    "logK_lo":   _bound_lo,   # None = no lower bound
                    "logK_hi":   _bound_hi,   # None = no upper bound
                })

            else:
                # Kinetic: parse all k_parts — may be "k1 = 1.0", "log k1 = 0.0", "k-1 = 2.0", …
                krate_pat = re.compile(
                    r"^(log\s+)?(k[A-Za-z0-9_\-]*)\s*=\s*(.+)$", re.IGNORECASE)
                parsed_ks = {}
                for kp in k_parts:
                    m = krate_pat.match(kp)
                    if not m:
                        raise ValueError(f"Cannot parse rate constant '{kp}' in: '{line}'")
                    is_log = m.group(1) is not None
                    kname  = m.group(2)
                    kval   = float(m.group(3))
                    log_k  = kval if is_log else math.log10(kval)
                    parsed_ks[kname] = log_k

                if not parsed_ks:
                    raise ValueError(f"No rate constants found in: '{line}'")

                knames_list = list(parsed_ks.keys())
                kname_fwd   = knames_list[0]
                entry = {
                    "type":      rxn_type,
                    "reactants": reactants,
                    "products":  products,
                    "kname":     kname_fwd,
                    "log_k":     parsed_ks[kname_fwd],
                }
                if rxn_type == "reversible_kinetic":
                    if len(knames_list) < 2:
                        raise ValueError(f"Reversible reaction '<>' needs k_fwd and k_rev: '{line}'")
                    kname_rev = knames_list[1]
                    entry["krname"] = kname_rev
                    entry["log_kr"] = parsed_ks[kname_rev]
                result["kinetics"].append(entry)

        elif section == "variables":
            m = re.match(r"(%?\w+)\s*=\s*(.+)", line)
            if m:
                varname = m.group(1).strip()
                expression = m.group(2).strip()
                result["variables"][varname] = expression

        elif section == "plot":
            m_xmax = re.match(rf"xmax\s*=\s*({num_re})", line)
            m_x    = re.match(r"x\s*=\s*(.+)",           line)
            m_y    = re.match(r"y\s*=\s*(.+)",           line)
            if m_xmax:
                result["plot_xmax"] = float(m_xmax.group(1))
            if m_x and not m_xmax:   # "x = expr" but not "xmax = ..."
                result["plot_x_expr"] = m_x.group(1).strip()
            if m_y:
                result["plot_y"] = [s.strip() for s in m_y.group(1).split(",") if s.strip()]

        elif section == "nmr":
            # "shift: Gtot"           -> fast-exchange chemical shift mode
            # "integration: 1, 3, 1"  -> slow-exchange integration mode
            # Both lines together     -> mixed slow+fast exchange mode
            m_shift = re.match(r"shift\s*:\s*(.+)", line, re.IGNORECASE)
            m_integ = re.match(r"integration\s*:\s*(.+)", line, re.IGNORECASE)
            if m_shift:
                targets = [t.strip() for t in m_shift.group(1).split(",") if t.strip()]
                if result["nmr"] is None:
                    result["nmr"] = {"mode": "shift", "targets": targets,
                                     "n_H_list": [], "n_integ": 0}
                else:
                    result["nmr"]["mode"]    = "mixed"
                    result["nmr"]["targets"] = targets
            elif m_integ:
                parts    = [t.strip() for t in m_integ.group(1).split(",") if t.strip()]
                n_H_list = []
                for p in parts:
                    try:    n_H_list.append(float(p))
                    except ValueError: pass
                if result["nmr"] is None:
                    result["nmr"] = {"mode": "integration", "n_H_list": n_H_list,
                                     "n_integ": len(n_H_list), "targets": []}
                else:
                    result["nmr"]["mode"]     = "mixed"
                    result["nmr"]["n_H_list"] = n_H_list
                    result["nmr"]["n_integ"]  = len(n_H_list)

        elif section == "spectra":
            # "transparent: H, X"  → species with zero absorbance everywhere
            # Also accept "transparent:" with no species (all species absorb)
            m_trans = re.match(r"transparent\s*:(.*)", line, re.IGNORECASE)
            if m_trans:
                raw = m_trans.group(1).strip()
                transparent = [t.strip() for t in raw.split(",") if t.strip()] if raw else []
                if result["spectra"] is None:
                    result["spectra"] = {"transparent": transparent}
                else:
                    result["spectra"]["transparent"] = transparent

    # ── Deduplicate knames across both equilibria and kinetics ───────────────
    from collections import Counter
    all_knames = ([eq["kname"] for eq in result["equilibria"]] +
                  [r["kname"]  for r in result["kinetics"]] +
                  [r["krname"] for r in result["kinetics"] if "krname" in r])
    kname_counts = Counter(all_knames)
    kname_seen   = Counter()
    for eq in result["equilibria"]:
        kn = eq["kname"]
        if kname_counts[kn] > 1:
            kname_seen[kn] += 1
            eq["kname"] = f"{kn}{kname_seen[kn]}"
    for rxn in result["kinetics"]:
        for attr in ("kname", "krname"):
            if attr in rxn:
                kn = rxn[attr]
                if kname_counts[kn] > 1:
                    kname_seen[kn] += 1
                    rxn[attr] = f"{kn}{kname_seen[kn]}"

    return result



def check_script_syntax(text: str) -> list:
    """
    Line-by-line syntax checker. Returns a list of (line_number, original_line, error_message).
    Does NOT raise — collects all errors so every problem is reported at once.
    """
    import re as _re

    VALID_SECTIONS = {
        "concentrations", "volumes", "titrant", "reactions", "equilibria",
        "variables", "plot", "nmr", "spectra",
    }
    CONC_UNITS  = {"m", "mm", "um", "µm"}
    VOL_UNITS   = {"l", "ml", "ul", "µl"}
    NUM_RE      = _re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")
    ARROW_RE    = _re.compile(r"(<>|>|=)")
    NAME_RE     = _re.compile(r"^[A-Za-z_][A-Za-z0-9_,]*$")

    errors  = []
    section = None

    # ── Cross-section tracking (populated during line pass) ──────────────
    defined_concs         = {}   # name -> (lineno, raw)
    defined_vols          = {}   # name -> (lineno, raw)
    defined_titrant       = {}   # name -> (lineno, raw)
    defined_vars          = {}   # name -> (lineno, raw)
    reaction_species      = set()
    reaction_species_refs = []   # (lineno, raw, name)
    plot_y_refs           = []   # (lineno, raw, name)
    plot_x_tokens         = []   # (lineno, raw, token)
    nmr_shift_refs        = []   # (lineno, raw, name)
    var_expr_refs         = []   # (lineno, raw, token)

    def err(lineno, raw, msg):
        errors.append((lineno, raw, msg))

    def is_number(s):
        return bool(NUM_RE.match(s.strip()))

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#")[0].strip()
        if not line:
            continue

        # ── Section headers ──────────────────────────────────────────
        if line.startswith("$"):
            sec_raw = line[1:].lower().strip()
            # $titrant or $titrant solid
            base = sec_raw.split()[0]
            if base == "titrant":
                section = "titrant"
            elif base in VALID_SECTIONS:
                section = base
                if "solid" in sec_raw and base != "titrant":
                    err(lineno, raw_line, f"'solid' modifier is only valid for $titrant, not ${base}.")
            else:
                err(lineno, raw_line, f"Unknown section '${sec_raw}'. Valid sections: {', '.join('$'+s for s in sorted(VALID_SECTIONS))}.")
                section = None
            continue

        if section is None:
            err(lineno, raw_line, "Line appears before any section header ($concentrations, $reactions, etc.).")
            continue

        # ── $concentrations ──────────────────────────────────────────
        if section == "concentrations":
            m = _re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if not m:
                err(lineno, raw_line, "Expected format: Name = value [unit]  (e.g. G0 = 1.00 mM).")
                continue
            defined_concs[m.group(1)] = (lineno, raw_line)
            if not m.group(1).endswith('0'):
                err(lineno, raw_line,
                    f"Concentration name '{m.group(1)}' should end in '0' to indicate an initial amount (e.g. '{m.group(1)}0').")
            val_str = m.group(2).strip()
            parts   = val_str.split()
            num_part = parts[0]
            unit_part = parts[1].lower() if len(parts) > 1 else "mm"
            if not is_number(num_part):
                err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
            elif unit_part not in CONC_UNITS:
                err(lineno, raw_line, f"Unknown concentration unit '{parts[1]}'. Use M, mM, or uM.")

        # ── $volumes ─────────────────────────────────────────────────
        elif section == "volumes":
            m = _re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if not m:
                err(lineno, raw_line, "Expected format: V0 = value [unit]  (e.g. V0 = 500 uL).")
                continue
            defined_vols[m.group(1)] = (lineno, raw_line)
            val_str   = m.group(2).strip()
            parts     = val_str.split()
            num_part  = parts[0]
            unit_part = parts[1].lower() if len(parts) > 1 else "ml"
            if not is_number(num_part):
                err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
            elif unit_part not in VOL_UNITS:
                err(lineno, raw_line, f"Unknown volume unit '{parts[1]}'. Use L, mL, or uL.")

        # ── $titrant ─────────────────────────────────────────────────
        elif section == "titrant":
            # Detect whether solid from the section header (stored externally);
            # here we just check the line format.
            parts = [p.strip() for p in line.split(";")]
            # Liquid: "Ht = 10.00 mM" or solid: "Ht" or "Ht; 1.3"
            if "=" in parts[0]:
                # liquid-mode line
                m = _re.match(r"^(\w+)\s*=\s*(.+)$", parts[0])
                if not m:
                    err(lineno, raw_line, "Expected format: Name = value [unit]  (e.g. Ht = 10.00 mM).")
                    continue
                defined_titrant[m.group(1)] = (lineno, raw_line)
                val_str   = m.group(2).strip()
                tparts    = val_str.split()
                num_part  = tparts[0]
                unit_part = tparts[1].lower() if len(tparts) > 1 else "mm"
                if not is_number(num_part):
                    err(lineno, raw_line, f"Cannot parse '{num_part}' as a number.")
                elif unit_part not in CONC_UNITS:
                    err(lineno, raw_line, f"Unknown concentration unit '{tparts[1]}'. Use M, mM, or uM.")
            else:
                # solid-mode line: just a name, optionally followed by "; ratio"
                name = parts[0]
                if not _re.match(r"^\w+$", name):
                    err(lineno, raw_line, f"Invalid titrant name '{name}'.")
                else:
                    defined_titrant[name] = (lineno, raw_line)
                if len(parts) > 1:
                    if not is_number(parts[1]):
                        err(lineno, raw_line, f"Stoichiometric ratio '{parts[1]}' is not a valid number.")

        # ── $reactions / $equilibria ──────────────────────────────────
        elif section in ("reactions", "equilibria"):
            # Must contain an arrow
            if not ARROW_RE.search(line):
                err(lineno, raw_line, "Reaction line must contain an arrow: = (equilibrium), > (irreversible), or <> (reversible).")
                continue
            # Split off the semicolon parameters
            main_part = line.split(";")[0].strip()
            arrow_m = ARROW_RE.search(main_part)
            if not arrow_m:
                err(lineno, raw_line, "Cannot find arrow in reaction left-hand/right-hand split.")
                continue
            arrow = arrow_m.group(0)
            arrow_pos = arrow_m.start()
            lhs = main_part[:arrow_pos].strip()
            rhs = main_part[arrow_pos + len(arrow):].strip()
            if not lhs:
                err(lineno, raw_line, "Reaction has no reactants (left side is empty).")
            if not rhs:
                err(lineno, raw_line, "Reaction has no products (right side is empty).")
            # Collect species names for cross-section checks
            for _side_str in (lhs, rhs):
                for _tok in _side_str.split("+"):
                    _tok = _tok.strip()
                    _m_sp = _re.match(r"^\d*\.?\d*\s*([A-Za-z_]\w*)$", _tok)
                    if _m_sp:
                        _sp = _m_sp.group(1)
                        reaction_species.add(_sp)
                        reaction_species_refs.append((lineno, raw_line, _sp))
            # Check for K/k parameter
            k_parts = [p.strip() for p in line.split(";")[1:]]
            if not k_parts:
                err(lineno, raw_line, "Reaction is missing equilibrium/rate constant (e.g. ; log K1 = 4.0).")
            else:
                # Validate first K/k part
                kp = k_parts[0]
                km = _re.match(r"^(log\s+)?([A-Za-z0-9_,\-]+)\s*=\s*(.+)$", kp, _re.IGNORECASE)
                if not km:
                    err(lineno, raw_line, f"Cannot parse constant '{kp}'. Expected: log Name = value  or  Name = value.")
                else:
                    val = km.group(3).strip()
                    if not is_number(val):
                        err(lineno, raw_line, f"Constant value '{val}' is not a valid number.")
                # Validate optional bounds: "from X to Y", "from X", "to Y"
                for extra in k_parts[1:]:
                    bm = _re.match(r"^(from\s+\S+\s+to\s+\S+|from\s+\S+|to\s+\S+)$", extra.strip(), _re.IGNORECASE)
                    if not bm:
                        # Could be a second K for reversible reaction — check
                        km2 = _re.match(r"^(log\s+)?([A-Za-z0-9_,\-]+)\s*=\s*(.+)$", extra, _re.IGNORECASE)
                        if not km2:
                            err(lineno, raw_line, f"Unrecognised parameter clause '{extra}'. Expected: log Name = value  or  from X to Y.")

        # ── $variables ────────────────────────────────────────────────
        elif section == "variables":
            m = _re.match(r"^(%?\w+)\s*=\s*(.+)$", line)
            if not m:
                err(lineno, raw_line, "Expected format: Name = expression  (e.g. Gtot = G + GH + GH2).")
            else:
                expr = m.group(2).strip()
                if not expr:
                    err(lineno, raw_line, "Variable expression is empty.")
                else:
                    defined_vars[m.group(1)] = (lineno, raw_line)
                    for _tok in (_re.findall(r"[A-Za-z_]\w*", expr) +
                                 _re.findall(r"%\w+", expr)):
                        var_expr_refs.append((lineno, raw_line, _tok))

        # ── $plot ─────────────────────────────────────────────────────
        elif section == "plot":
            m = _re.match(r"^(xmax|x|y)\s*=\s*(.+)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line, "Expected one of: xmax = value, x = expression, y = species1, species2, …")
            else:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == "xmax" and not is_number(val):
                    err(lineno, raw_line, f"xmax value '{val}' is not a valid number.")
                elif key == "y":
                    for _name in [n.strip() for n in val.split(",") if n.strip()]:
                        plot_y_refs.append((lineno, raw_line, _name))
                        # Also register %name in defined_vars if not already there
                        # so the cross-section check doesn't flag it as undefined
                elif key == "x":
                    for _tok in _re.findall(r"[A-Za-z_]\w*", val):
                        plot_x_tokens.append((lineno, raw_line, _tok))

        # ── $nmr ─────────────────────────────────────────────────────
        elif section == "nmr":
            m = _re.match(r"^(shift|integration)\s*:\s*(.*)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line, "Expected: shift: TargetName  or  integration: n1, n2, …")
            else:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key == "integration" and val:
                    for tok in [t.strip() for t in val.split(",") if t.strip()]:
                        if not is_number(tok):
                            err(lineno, raw_line, f"integration: expects numbers separated by commas; '{tok}' is not a valid number.")
                elif key == "shift" and val:
                    for _name in [t.strip() for t in val.split(",") if t.strip()]:
                        nmr_shift_refs.append((lineno, raw_line, _name))

        # ── $spectra ──────────────────────────────────────────────────
        elif section == "spectra":
            m = _re.match(r"^transparent\s*:\s*(.*)$", line, _re.IGNORECASE)
            if not m:
                err(lineno, raw_line, "Only 'transparent: Species1, Species2, …' is valid inside $spectra (or leave the section body empty).")

    # ── Cross-section validation ──────────────────────────────────────────
    # Each 'X0' in $concentrations implicitly defines 'X' as a free species
    implicit_species = {name[:-1] for name in defined_concs if name.endswith('0') and len(name) > 1}
    valid_names = (reaction_species
                   | set(defined_concs)
                   | set(defined_titrant)
                   | set(defined_vars)
                   | implicit_species)
    vol_names   = set(defined_vols)

    # Concentrations defined but never used in any reaction, plot x, plot y, or variable expression
    # Also count usage of the implicit species (e.g. 'G' counts as usage of 'G0')
    _all_refs = (reaction_species
                 | {tok for _, _, tok in plot_x_tokens}
                 | {tok for _, _, tok in plot_y_refs}
                 | {tok for _, _, tok in var_expr_refs}
                 | {tok for _, _, tok in nmr_shift_refs})
    for _name in defined_concs:
        _implicit = _name[:-1] if _name.endswith('0') and len(_name) > 1 else None
        if _name not in _all_refs and (_implicit is None or _implicit not in _all_refs):
            _ln, _raw = defined_concs[_name]
            err(_ln, _raw,
                f"'{_name}' is defined in $concentrations but never appears in any reaction or expression. Possible typo?")

    # Volume names must not appear as reaction species
    for _ln, _raw, _sp in reaction_species_refs:
        if _sp in vol_names:
            err(_ln, _raw,
                f"'{_sp}' is defined as a volume in $volumes and cannot be used as a chemical species.")

    # All names in $plot y must be defined
    for _ln, _raw, _name in plot_y_refs:
        if _name not in valid_names:
            err(_ln, _raw,
                f"'{_name}' in '$plot y' is not defined as a species, concentration, titrant, or variable.")

    # All identifier tokens in $plot x expression must be defined
    for _ln, _raw, _tok in plot_x_tokens:
        if _tok in vol_names:
            err(_ln, _raw,
                f"'{_tok}' is a volume ($volumes) and cannot be used in a '$plot x' expression.")
        elif _tok not in valid_names:
            err(_ln, _raw,
                f"'{_tok}' in '$plot x' expression is not defined as a species, concentration, titrant, or variable.")

    # All targets in $nmr shift must be defined
    for _ln, _raw, _name in nmr_shift_refs:
        if _name not in valid_names:
            err(_ln, _raw,
                f"'{_name}' in '$nmr shift' is not defined as a species, concentration, or variable.")

    # All identifier tokens in $variables expressions must be defined
    for _ln, _raw, _tok in var_expr_refs:
        if _tok in vol_names:
            err(_ln, _raw,
                f"'{_tok}' is a volume ($volumes) and cannot be used in a '$variables' expression.")
        elif _tok not in valid_names:
            err(_ln, _raw,
                f"'{_tok}' in '$variables' expression is not defined as a species, concentration, titrant, or variable.")

    return errors
