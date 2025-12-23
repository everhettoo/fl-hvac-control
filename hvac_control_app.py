import numpy as np
""" Membership Functions """
from mylibs.triangular_mem import tri
from mylibs.trapeziod_mem import trap

""" Universe of Discourse """
# input variables
temp = np.linspace(18, 30, 400)
humid = np.linspace(25, 85, 400)
co2 = np.linspace(300, 1600, 500)
# output variables
cooling = np.linspace(0, 100, 400)

""" Fuzzy Sets Definition """
# Indoor Temperature
def temp_low(x):
    return trap(x, 18, 18.5, 21.5, 22.5)
def temp_med(x):
    return tri(x, 21, 23.5, 26)
def temp_high(x):
    return trap(x, 25, 26, 29, 40)

# Indoor Humidity
def humid_low(x):
    return trap(x, 25, 30, 40, 45)
def humid_med(x):
    return tri(x, 40, 52.5, 65)
def humid_high(x):
    return trap(x, 60, 65, 80, 85)

# CO2 Concentration
def co2_low(x):
    return trap(x, 300, 400, 600, 700)
def co2_med(x):
    return tri(x, 600, 850, 1100)
def co2_high(x):
    return trap(x, 1000, 1150, 1450, 1600)

# Cooling levels
def cooling_off(x):
    return np.array([trap(val, 0, 0, 15, 25) for val in np.atleast_1d(x)])
def cooling_low(x):
    return np.array([tri(x, 20, 25, 40) for x in np.atleast_1d(x)])
def cooling_med(x):
    return np.array([tri(x, 40, 60, 70) for x in np.atleast_1d(x)])
def cooling_high(x):
    return np.array([trap(x, 70, 80, 100, 100) for x in np.atleast_1d(x)])

""" Rules Evaluation and Defuzzification """
def evaluate_rules(temp_val, humid_val, co2_val, universe):
    # Fuzzify inputs (scalar membership degrees)
    T_low = temp_low(temp_val)
    T_med = temp_med(temp_val)
    T_high = temp_high(temp_val)

    H_low = humid_low(humid_val)
    H_med = humid_med(humid_val)
    H_high = humid_high(humid_val)

    C_low = co2_low(co2_val)
    C_med = co2_med(co2_val)
    C_high = co2_high(co2_val)

    # Rule outputs (apply min for AND, clip output membership functions)
    rule1 = np.minimum(C_low, cooling_off(universe))                  # CO2 Low → AC Off
    rule2 = np.minimum(np.minimum(C_med, T_med), cooling_low(universe))   # CO2 Med & Temp Comfortable → AC Low
    rule3 = np.minimum(np.minimum(C_med, T_high), cooling_med(universe))  # CO2 Med & Temp Warm → AC Medium
    rule4 = np.minimum(np.minimum(C_high, T_med), cooling_high(universe)) # CO2 High & Temp Comfortable → AC High
    rule5 = np.minimum(np.minimum(C_high, H_high), cooling_high(universe))# CO2 High & Humid High → AC High
    rule6 = np.minimum(np.minimum(C_med, H_high), cooling_med(universe))  # CO2 Med & Humid High → AC Medium
    rule7 = np.minimum(np.minimum(C_low, T_high), cooling_low(universe))  # CO2 Low & Temp Warm → AC Low

    # Aggregate all rule outputs (apply max for OR)
    aggregated = np.maximum.reduce([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    return aggregated

""" Defuzzification (Centroid) """
def defuzzify(universe, aggregated):
    if np.sum(aggregated) == 0:
        return 0.0
    return np.sum(universe * aggregated) / np.sum(aggregated)

""" Input Validation """
def validate_input(value, universe, name):
    if value < universe.min() or value > universe.max():
        raise ValueError(f"{name} value {value} is out of universe range ({universe.min()}–{universe.max()})")

if __name__ == "__main__":
    print("HVAC Control System using Fuzzy Logic")

    # Sample input values
    input_temp = 30.0   # Current indoor temperature in °C
    input_humid = 55.0  # Current indoor humidity in %
    input_co2 = 900.0   # Current CO2 concentration in ppm

    # Validate inputs
    validate_input(input_temp, temp, "Temperature")
    validate_input(input_humid, humid, "Humidity")
    validate_input(input_co2, co2, "CO₂")

    # Fuzzify inputs
    temp_memberships = {
        "LOW": temp_low(input_temp),
        "MEDIUM": temp_med(input_temp),
        "HIGH": temp_high(input_temp)
    }
    humid_memberships = {
        "LOW": humid_low(input_humid),
        "MEDIUM": humid_med(input_humid),
        "HIGH": humid_high(input_humid)
    }
    co2_memberships = {
        "LOW": co2_low(input_co2),
        "MEDIUM": co2_med(input_co2),
        "HIGH": co2_high(input_co2)
    }
    temp_category = max(temp_memberships, key=temp_memberships.get)
    humid_category = max(humid_memberships, key=humid_memberships.get)
    co2_category = max(co2_memberships, key=co2_memberships.get)

    # Print inputs with categories
    print(f"Input Temperature: {input_temp:.1f} °C → {temp_category}")
    print(f"Input Humidity: {input_humid:.1f} % → {humid_category}")
    print(f"Input CO₂: {input_co2:.1f} ppm → {co2_category}")

    # Rule evaluation
    aggregated = evaluate_rules(input_temp, input_humid, input_co2, cooling)

    # Defuzzification
    cooling_level = defuzzify(cooling, aggregated)

    # Membership degrees of the crisp output
    mu_off = cooling_off(cooling_level)[0]
    mu_low = cooling_low(cooling_level)[0]
    mu_med = cooling_med(cooling_level)[0]
    mu_high = cooling_high(cooling_level)[0]

    # Put them in a dictionary for easy handling
    memberships = {
        "OFF": mu_off,
        "LOW": mu_low,
        "MEDIUM": mu_med,
        "HIGH": mu_high
    }

    # Find the fuzzy set with the highest membership
    dominant_label = max(memberships, key=memberships.get)

    print(f"Recommended Cooling Level: {cooling_level:.2f} %")
    print(f"👉 Dominant Fuzzy Set: {dominant_label}")