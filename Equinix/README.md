# Equinix

Fitting software for equilibrium titrations and kinetics experiments.  
Supports UV-Vis, NMR, and concentration data. Runs entirely on your computer — no data is sent anywhere.

---

## Requirements

- Python 3.10 or later **or** Miniconda (see below)
- All other dependencies are installed automatically

---

## Installation — choose one method

### Option A: conda (recommended, especially on Windows)

Best if you already have Anaconda or Miniconda, or if Option B gives you trouble.

1. Install Miniconda if needed: https://docs.conda.io/en/latest/miniconda.html
2. Open a terminal (Mac/Linux) or Anaconda Prompt (Windows)
3. Navigate to the Equinix folder:
   ```
   cd path/to/Equinix
   ```
4. Create the environment (once):
   ```
   conda env create -f environment.yml
   ```
5. Activate it:
   ```
   conda activate equinix
   ```

### Option B: pip

Best if you already have Python 3.10+ installed.

1. Open a terminal and navigate to the Equinix folder:
   ```
   cd path/to/Equinix
   ```
2. Install dependencies (once):
   ```
   pip install -r requirements.txt
   ```

---

## Running the app

After installing (Option A or B), launch the app with:
```
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

**Note for conda users:** make sure the environment is activated first (`conda activate equinix`).

---

## Every time you use Equinix

```
conda activate equinix        ← conda users only
streamlit run app.py
```

---

## Troubleshooting

**"streamlit: command not found"**  
Run `pip install streamlit` or make sure the conda environment is activated.

**Publication figure (PDF) button does nothing**  
Run `pip install kaleido==0.2.1` and restart the app.

**Wrong Python version**  
Make sure you are using Python 3.10 or later: `python --version`

---

## Batch processing (optional)

To fit and export all scripts in a folder at once:
```
python batch_process.py path/to/scripts/ path/to/output/
```

## Running tests (optional)

```
python run_tests.py path/to/scripts/
```
