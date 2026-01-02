# Fuzzy Logic HVAC Control
A simple HVAC controller implemented with fuzzy logic (Mamdani-style). The project contains notebooks to explore membership functions and a small Python application that demonstrates the controller with plots and printed summaries.

## Prerequisites
- Python 3.8+ (development used Python 3.12.9). Ensure Python is on your PATH.
- (Optional but recommended) Create and use a virtual environment to avoid polluting the system Python.

## Quick setup
From the repository root (Windows PowerShell):

```powershell
# Create and activate a venv (Windows PowerShell)
python -m venv venv; .\venv\Scripts\Activate.ps1
# Install required packages
pip install -r requirements.txt
```

If you're using cmd.exe instead of PowerShell, activate the venv with:

```powershell
venv\Scripts\activate
```

## Project structure
- `main.py` — Example script to run the HVAC controller and produce plots and a printed summary.
- `mylibs/` — Package with membership functions and helper utilities:
  - `mylibs/membership_functions.py` — increasing/decreasing/triangular/trapezoidal/gaussian/sigmoid membership functions and defuzzification helpers.
- `notebooks/` — Jupyter notebooks used for exploration and verification:
  - `1_explore_temperature_membership.ipynb` — explore and plot temperature membership functions and edge cases.
  - `2_explore_humidity_membership.ipynb` — explore humidity membership functions and shapes.
  - `3_explore_co2_membership.ipynb` — explore CO₂ membership functions and thresholds.
  - `4_explore_hvac_membership.ipynb` — defines the HVAC output membership functions and visualizes them.
  - `5_mamdani_hvac_control.ipynb` — step-by-step Mamdani controller setup and rule visualization.
  - `6_hvac_verification.ipynb` — verification and sample runs comparing fuzzy output to expected behaviour.
- `tests/` — Unit tests for the membership functions (pytest).
- `requirements.txt` — Python package dependencies.

## Run: for observation
This section gives a few practical ways to run the example application and observe results (plots + printed summary).

1) Run the interactive example script (recommended)

Open PowerShell from the repository root (venv activated) and run:

```powershell
python main.py
```

Expected behaviour:
- The script prints input summaries and dominant categories. Example lines might look like:
  - "HVAC Control System using Fuzzy Logic"
  - "Input Temperature: 21.5 °C"
  - "Temperature Category: Comfortable (μ=0.75)"
  - "Recommended HVAC Level: 42.00 % (Medium)"
- Matplotlib windows will open showing fuzzification of inputs and the defuzzified HVAC output.

2) Run with custom scalar inputs (no code edit required)

You can call the main function from a one-liner in PowerShell:

```powershell
python -c "from main import hvac_control_app; hvac_control_app(24.0, 50.0, 900.0)"
```

This runs the app for temperature=24.0 °C, humidity=50.0 %, CO₂=900 ppm and will produce the same printed output and plots.

## Testing
Run the unit tests with pytest from the repository root (venv activated):

```powershell
pip install -r requirements.txt; pytest -q
```

The `tests/` directory contains tests for the membership function implementations.

## Troubleshooting
- Module import errors: Ensure you run commands from the repository root so Python finds the `mylibs` package, and that the virtual environment is activated.
- Missing packages: Verify installation with `pip list` and `pip install -r requirements.txt`.
- Plots not showing: On some remote or headless systems, matplotlib requires a non-interactive backend; see matplotlib docs or comment out `plt.show()` in `main.py`.

## Contributing
Contributions are welcome. If you change public behavior (APIs, function signatures) please add or update tests in `tests/`.

## License
This project is licensed under the terms in the `LICENSE` file.

Enjoy exploring the fuzzy HVAC controller! Feel free to modify input values and observe how the system responds.
