# Equilibrist

A browser-based thermodynamic and kinetic solver built with Python and Streamlit.  
Define a chemical system in a plain-text script, simulate titration or time-resolved curves, and fit binding or rate constants to experimental data.  
Runs entirely on your local machine — no data is sent anywhere.  
**Version 2.0** adds a complete statistical-validation, parameter-identifiability and uncertainty-quantification layer.

---

## Capabilities

- **Equilibrium titrations** — arbitrary reaction networks, any stoichiometry, multiple products per reaction
- **Kinetics simulations** — irreversible (`>`) and reversible (`<>`) steps, mixed equilibrium/kinetics scripts
- **Free-energy input** — reactions defined via ΔG° (van 't Hoff) and ΔG‡ (Eyring) in kcal/mol
- **Acid-base mode** — `$reactions acid-base` accepts pKa values directly; water equilibrium, water concentration, and a `pH` variable are injected automatically; ladder syntax for polyprotic acids
- **Solution and solid titrant modes** — volume-based dilution, or solid addition with fixed cell volume
- **UV-Vis global spectral fitting** — Beer–Lambert analysis; molar absorptivities solved analytically (variable projection); known spectra for individual species can be provided in a second sheet of the data file via `read:` in `$spectra`, pinning those species' ε(λ) during the fit
- **NMR fitting** — fast-exchange chemical shift, slow-exchange integration, and simultaneous slow + fast mixed mode; all three available in both equilibrium and kinetics modes. `read:` in `$nmr` pins per-column intrinsic shifts from a second sheet of the data file; `noref` releases the auto-pinned reference so an externally anchored absolute shift scale is recovered when `read:` is also present
- **Concentration fitting** — direct fitting of measured species concentrations (mM)
- **Dual optimizer** — L-BFGS-B warm-start followed by Nelder-Mead; user-selectable per run
- **Thermodynamic constraints** — inter-parameter relationships among equilibrium and rate constants enforced as soft or hard penalties; v2.0 adds NMR shift constraints (equality groups `A = B = C`, monotone chains `A > B > C`, per-column magnitude bounds) for fast-exchange fits
- **Thermodynamic cycle detection** — automatic warning on inconsistent K values
- **Outlier flagging** — click any experimental data point to exclude it from fitting; excluded points are absent from all exported figures
- **Statistical validation** — AIC, AICc and BIC printed with every fit; Akaike weights and F-test for nested model comparison; Durbin–Watson and Shapiro–Wilk residual tests; four-panel residual figure (residuals vs predictor, histogram, normal Q–Q, metrics)
- **Identifiability tools** — 1D and 2D RMSE profiles with Hessian-derived confidence ellipses overlaid; parameter correlation heatmap; sloppy-spectrum eigendecomposition with stiff/sloppy ranking and per-parameter impact bars; per-parameter *t*-tests with user-editable null hypotheses
- **Uncertainty quantification** — non-parametric residual bootstrap, parametric Monte Carlo, and Mammen wild bootstrap; leave-one-out jackknife; Monte Carlo propagation of user-specified relative uncertainty on stock concentrations, titrant stocks and V₀. All three layers are available for every fit mode and can run in parallel across all cores
- **Local sensitivity (Masson ξ)** — 3<sup>N</sup> paired-difference probe over arbitrary subsets of fitted parameters and held-fixed quantities, returning conditional and marginal standard errors and a coupling ratio
- **FAIR session save/restore** — single self-contained JSON bundle (script + experimental data with SHA-256 checksum + fit state + every uncertainty and sensitivity result + RNG seed + Python / NumPy / SciPy / joblib versions). Drop the `.json` back into the script uploader to reconstruct the analysis exactly
- **Excel export** — timestamped workbook with data, script, and parameters tabs
- **Publication-quality PDF export** — vector text, Arial 9 pt, inward ticks, editable in Illustrator
- **In-app script editor** — edit and re-run without leaving the browser

---

## Requirements

- Python 3.10 or later **or** Miniconda (see below)
- All other dependencies are installed automatically from `requirements.txt` or `environment.yml`

---

## Installation

### Option A — conda (recommended, especially on Windows)

Best if you already have Anaconda or Miniconda installed, or if Option B gives you trouble.  
If you need Miniconda: https://docs.conda.io/en/latest/miniconda.html

```
cd path/to/Equilibrist
conda env create -f environment.yml
conda activate equilibrist
```

### Option B — pip

Best if you already have Python 3.10+ installed.

```
cd path/to/Equilibrist
pip install -r requirements.txt
```

---

## Running the app

```
conda activate equilibrist    # conda users only — skip if using pip
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.  
Press **Ctrl + C** in the terminal to stop the server.

---

## File layout

```
Equilibrist/
├── app.py                              ← main application entry point
├── equilibrist_parser.py
├── equilibrist_network.py
├── equilibrist_kinetics.py
├── equilibrist_kinetics_nmr.py
├── equilibrist_kinetics_spectra.py
├── equilibrist_curve.py
├── equilibrist_fit_conc.py
├── equilibrist_fit_nmr.py
├── equilibrist_fit_spectra.py
├── equilibrist_shift_constraints.py    ← v2.0: NMR shift-constraint solver
├── equilibrist_diagnostics.py          ← v2.0: statistical & identifiability layer
├── equilibrist_bootstrap.py            ← v2.0: bootstrap / jackknife / Monte Carlo
├── equilibrist_session.py              ← v2.0: FAIR JSON session save/restore
├── equilibrist_io.py
├── requirements.txt                    ← pip dependency list
├── environment.yml                     ← conda environment definition
├── README.md
└── Equilibrist_manual.html             ← full user manual (open in any browser)
```

A `__pycache__` folder may appear after the first run — this is normal.

---

## Troubleshooting

**"streamlit: command not found"**  
Run `pip install streamlit`, or make sure the conda environment is activated first.

**"pip is not recognized" (Windows)**  
Use `python -m pip install -r requirements.txt` instead.

**Wrong Python version**  
Verify with `python --version` — Equilibrist requires 3.10 or later.

**App won't start after updating files**  
Press **Ctrl + C** to stop any running instance, then relaunch with `streamlit run app.py`.

---

## Documentation

Open `Equilibrist_manual.html` in any browser for the full user manual, including all script syntax, data file formats, fitting modes, the complete statistical-validation and uncertainty-quantification layer, and three new worked examples (residual diagnostics on a 1:1 binding, 2D RMSE profile + bootstrap + jackknife + local sensitivity on a 2:1 NMR shift fit, and correlation heatmap + sloppy spectrum + Monte Carlo on a kinetics + UV-Vis fit) that mirror the case studies in the accompanying paper.

---

*Equilibrist v2.0 — built with Python & Streamlit · all computation local, no data transmitted*  
*© Eric Masson, Ohio University, 2026*
