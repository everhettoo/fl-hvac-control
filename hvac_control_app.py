import numpy as np
import mylibs.membership_functions as mf

""" Universe of Discourse """
# input variables
temp = np.linspace(18, 30, 400)
humid = np.linspace(25, 85, 400)
co2 = np.linspace(300, 1600, 500)
# output variables
cooling = np.linspace(0, 100, 400)


""" Membership Design """
temp = np.linspace(16, 30, 400)
cold_temp = np.array([mf.trap(x, 16, 16, 18,22) for x in temp])
comfortable_temp = np.array([mf.tri(x, 20, 23, 26) for x in temp])
warm_temp = np.array([mf.trap(x, 24, 27, 30, 30) for x in temp])

humid = np.linspace(30, 80, 400)
dry_humid = np.array([mf.trap(x, 30, 30, 35, 45) for x in humid])
normal_humid = np.array([mf.tri(x, 40, 55, 70) for x in humid])
high_humid = np.array([mf.trap(x, 60, 70, 80, 80) for x in humid])

co2 = np.linspace(400, 1500, 400)
low_co2 = np.array([mf.trap(x, 400, 400, 600, 800) for x in co2])
medium_co2 = np.array([mf.tri(x, 700, 950, 1200) for x in co2])
high_co2 = np.array([mf.trap(x, 1000, 1200, 1500, 1500) for x in co2])

cooling = np.linspace(0, 100, 400)
off_cool = np.array([mf.trap(x, 0, 0, 5, 15) for x in cooling])
low_cool = np.array([mf.tri(x, 10, 25, 40) for x in cooling])
medium_cool = np.array([mf.tri(x, 35, 55, 75) for x in cooling])
high_cool = np.array([mf.trap(x, 70, 85, 100, 100) for x in cooling])


""" Fuzzification Membership Functions """
in_cold_temp = 0
in_comfortable_temp = 0
in_warm_temp = 0

def fuzzify_temp(x):
    global in_cold_temp, in_comfortable_temp, in_warm_temp
    in_cold_temp = mf.trap(x, 16, 16, 18, 22)
    in_comfortable_temp = mf.tri(x, 20, 23, 26)
    in_warm_temp = mf.trap(x, 24, 27, 30, 30)

in_dry_humid = 0
in_normal_humid = 0
in_high_humid = 0

def fuzzify_humid(x):
    global in_dry_humid, in_normal_humid, in_high_humid
    in_dry_humid = mf.trap(x, 30, 30, 35, 45)
    in_normal_humid = mf.tri(x, 40, 55, 70)
    in_high_humid = mf.trap(x, 60, 70, 80, 80)

in_low_co2 = 0
in_medium_co2 = 0
in_high_co2 = 0

def fuzzify_co2(x):
    global in_low_co2, in_medium_co2, in_high_co2
    in_low_co2 = mf.trap(x, 400, 400, 600, 800)
    in_medium_co2 = mf.tri(x, 700, 950, 1200)
    in_high_co2 = mf.trap(x, 1000, 1200, 1500, 1500)


""" Rules Evaluation and Defuzzification """
def evaluate_rules():
    # R1 : If Temp is Cold, THEN Cooling is Off
    R1 = np.fmin(in_cold_temp, off_cool)

    # R2 : If Humidity is high and Temp is Warm, THEN Cooling is High
    ant = np.min([in_high_humid, in_warm_temp])
    R2 = np.fmin(ant, high_cool)

    # R3 : If CO2 is High and Temp is Warm, THEN Cooling is High
    ant = np.min([in_high_co2, in_warm_temp])
    R3 = np.fmin(ant, high_cool)

    # R4 : If CO2 is Medium and Temp is Warm, THEN Cooling is Medium
    ant = np.min([in_medium_co2, in_warm_temp])
    R4 = np.fmin(ant, medium_cool)

    # R5 : If CO2 is Medium and Temp is Comfortable, THEN Cooling is Low
    ant = np.min([in_medium_co2, in_comfortable_temp])
    R5 = np.fmin(ant, low_cool)

    # R6 : If CO2 is Low and Temp is Warm, THEN Cooling is Low
    ant = np.min([in_low_co2, in_warm_temp])
    R6 = np.fmin(ant, low_cool)

    # R7 : If CO2 is Low and Temp is Comfortable and Humidity is Normal, THEN Cooling is Medium
    ant = np.min([in_low_co2, in_comfortable_temp, in_normal_humid])
    R7 = np.fmin(ant, off_cool)

    return np.fmax(R1, np.fmax(R2, np.fmax(R3, np.fmax(R4, np.fmax(R5, np.fmax(R6, R7))))))


""" Defuzzification (Centroid) """
def defuzzify(universe, aggregated):
    if np.sum(aggregated) == 0:
        return 0.0
    return np.sum(universe * aggregated) / np.sum(aggregated)

def dominant_category(name, memberships):
    # memberships is a dict {category: value}
    dominant = max(memberships, key=memberships.get)
    print(f"{name} Category: {dominant} (μ={memberships[dominant]:.2f})")
    return dominant

if __name__ == "__main__":
    print("HVAC Control System using Fuzzy Logic")

    # Sample input values
    in_temp = 21.5   # Current indoor temperature in °C
    in_humid = 42  # Current indoor humidity in %
    in_co2 = 600   # Current CO2 concentration in ppm

    # Print inputs with categories
    print(f"Input Temperature: {in_temp:.1f} °C")
    print(f"Input Humidity: {in_humid:.1f}")
    print(f"Input CO₂: {in_co2:.1f} ppm")

    # Fuzzification
    fuzzify_temp(in_temp)
    fuzzify_humid(in_humid)
    fuzzify_co2(in_co2)

    # Collect memberships
    temp_memberships = {
        "Cold": in_cold_temp,
        "Comfortable": in_comfortable_temp,
        "Warm": in_warm_temp
    }
    humid_memberships = {
        "Dry": in_dry_humid,
        "Normal": in_normal_humid,
        "High": in_high_humid
    }
    co2_memberships = {
        "Low": in_low_co2,
        "Medium": in_medium_co2,
        "High": in_high_co2
    }

    # Show dominant categories
    temp_cat = dominant_category("Temperature", temp_memberships)
    humid_cat = dominant_category("Humidity", humid_memberships)
    co2_cat = dominant_category("CO₂", co2_memberships)

    # Rules Evaluation
    aggregated = evaluate_rules()

    # Defuzzification
    cooling_level = defuzzify(cooling, aggregated)

    print(f"Recommended Cooling Level: {cooling_level:.2f} %")

    # Cooling category (compare crisp output to membership functions)
    cooling_memberships = {
        "Off": mf.trap(cooling_level, 0, 0, 5, 15),
        "Low": mf.tri(cooling_level, 10, 25, 40),
        "Medium": mf.tri(cooling_level, 35, 55, 75),
        "High": mf.trap(cooling_level, 70, 85, 100, 100)
    }

    cool_cat = dominant_category("Cooling", cooling_memberships)

    print(f"Recommended Cooling Level: {cooling_level:.2f} % ({cool_cat})")