# Fuzzy Logic HVAC Control
A simple fuzzy logic HVAC controller implementation using Mamdani model. The project contains notebooks to explore membership functions and a small Python application that demonstrates the controller with plots and printed summaries.

## Project structure
The project contains the following structure.
- `main.py` — Example script to run the HVAC controller and produce plots and a printed summary.
- `mylibs/` — Package with membership functions and helper utilities:
  - `mylibs/membership_functions.py` — increasing/decreasing/triangular/trapezoidal/gaussian/sigmoid membership functions and defuzzification helpers.
- `notebooks/` — Jupyter notebooks used for exploration and verification:
  - `1_explore_temperature_membership.ipynb` — explore and plot temperature membership functions and edge cases.
  - `2_explore_humidity_membership.ipynb` — explore humidity membership functions and shapes.
  - `3_explore_co2_membership.ipynb` — explore CO₂ membership functions and thresholds.
  - `4_explore_hvac_membership.ipynb` — defines the HVAC output membership functions and visualizes them.
  - `5_mamdani_hvac_control.ipynb` — a Mamdani FLS development: membership function definition, fuzzification, rule-evaluation, and defuzzification..
  - `6_hvac_verification.ipynb` — includes the tests for: optimized rules, overlapping membership, and exception cases..
- `tests/` — Unit tests for the membership functions.
- `requirements.txt` — Python package dependencies.

## Prerequisites
- Python 3.8+ (development used Python 3.12.9). Ensure Python is on your PATH.

## Clone & setup
From the repository root:

1.	Clone the project to local: ```git clone git@github.com:everhettoo/fl-hvac-control.git```
2.	Open the project using the preferred IDE (PyCharm or VSCode)
3.	Create a virtual environment (if needed): ```python -m venv venv```
4.	Activate the virtual environment: ```source venv/bin/activate``` (bash), or ```venv/Scripts/activate``` (Windows)
5.	Install required packages: ```pip install -r requirements.txt```

## Two parts to Run:
1. Jupyter Notebooks - to observe the development approach as explained earlier (run all notebooks for better understanding).
2. HVAC Application - to run the HVAC controller python script.

## Run: Jupyter Notebook
1. Start jupyter server from the same terminal: ```jupyter notebook```
2. Browse notebooks from browser: http://localhost:8888/tree

## Run: Application
```python main.py 21.5 55 600```

Expected behaviour:
- The script prints input summaries and dominant categories. Example lines might look like:
  - "HVAC Control System using Fuzzy Logic"
  - "Input Temperature: 21.5 °C"
  - "Temperature Category: Comfortable (μ=0.75)"
  - "Recommended HVAC Level: 42.00 % (Medium)"
- Matplotlib windows will open showing fuzzification of inputs and the defuzzified HVAC output.

## Testing
The `tests/` directory contains tests for the membership function implementations.

```python -m unittest discover -s tests```

## Troubleshooting
- Module import errors: Ensure you run commands from the repository root so Python finds the `mylibs` package, and that the virtual environment is activated.
- Missing packages: Verify installation with `pip list` and `pip install -r requirements.txt`.
- Plots not showing: On some remote or headless systems, matplotlib requires a non-interactive backend; see matplotlib docs or comment out `plt.show()` in `main.py`.

## Contributing
Contributions are welcome. If you change public behavior (APIs, function signatures) please add or update tests in `tests/`.

## License
This project is licensed under the terms in the `LICENSE` file.

Enjoy exploring the fuzzy HVAC controller! Feel free to modify input values and observe how the system responds.
