import numpy as np
from matplotlib import pyplot as plt

import mylibs.membership_functions as mf

""" Universe of Discourse """
# input variables
temp = np.linspace(18, 30, 400)
humid = np.linspace(25, 85, 400)
co2 = np.linspace(300, 1600, 500)
# output variables
hvac = np.linspace(0, 100, 400)


""" Membership Design """
cold_temp = np.array([mf.trap(x, 18, 18, 20, 22) for x in temp])
comfortable_temp = np.array([mf.tri(x, 20, 23.5, 27) for x in temp])
warm_temp = np.array([mf.trap(x, 25, 27, 30, 30) for x in temp])

dry_humid = np.array([mf.trap(x, 30, 30, 37, 45) for x in humid])
normal_humid = np.array([mf.tri(x, 40, 52.5, 65) for x in humid])
high_humid = np.array([mf.trap(x, 60, 70, 80, 80) for x in humid])

low_co2 = np.array([mf.trap(x, 400, 400, 500, 600) for x in co2])
medium_co2 = np.array([mf.tri(x, 500, 800, 1100) for x in co2])
high_co2 = np.array([mf.trap(x, 1000, 1250, 1500, 1500) for x in co2])

off_hvac = np.array([mf.trap(x, 0, 0, 5, 15) for x in hvac])
low_hvac = np.array([mf.tri(x, 10, 25, 40) for x in hvac])
medium_hvac = np.array([mf.tri(x, 35, 55, 75) for x in hvac])
high_hvac = np.array([mf.trap(x, 70, 85, 100, 100) for x in hvac])


""" Fuzzification Membership Functions """
in_cold_temp = 0
in_comfortable_temp = 0
in_warm_temp = 0


def fuzzify_temp(x):
    global in_cold_temp, in_comfortable_temp, in_warm_temp
    in_cold_temp = mf.trap(x, 18, 18, 20, 22)  # take middle point of the trap
    in_comfortable_temp = mf.tri(x, 20, 23.5, 27)  # take middle point of the tri
    in_warm_temp = mf.trap(x, 25, 27, 30, 30)  # take middle point of the trap


in_dry_humid = 0
in_normal_humid = 0
in_high_humid = 0


def fuzzify_humid(x):
    global in_dry_humid, in_normal_humid, in_high_humid
    in_dry_humid = mf.trap(x, 30, 30, 37, 45)  # take middle point of the trap
    in_normal_humid = mf.tri(x, 40, 52.5, 65)  # take middle point of the tri
    in_high_humid = mf.trap(x, 60, 70, 80, 80)  # take middle point of the trap


in_low_co2 = 0
in_medium_co2 = 0
in_high_co2 = 0


def fuzzify_co2(x):
    global in_low_co2, in_medium_co2, in_high_co2
    in_low_co2 = mf.trap(x, 400, 400, 500, 600)  # take middle point of the trap
    in_medium_co2 = mf.tri(x, 500, 800, 1100)  # take middle point of the tri
    in_high_co2 = mf.trap(x, 1000, 1250, 1500, 1500)  # take middle point of the trap


""" Rules Evaluation and Defuzzification """


def evaluate_rules():
    antecedent_1 = min(in_comfortable_temp, in_normal_humid, in_low_co2)
    antecedent_2 = min(in_comfortable_temp, in_normal_humid, in_medium_co2)
    antecedent_3 = min(in_cold_temp, in_normal_humid)
    antecedent_4 = min(in_warm_temp, in_normal_humid)
    antecedent_5 = in_high_humid
    antecedent_6 = in_high_co2
    antecedent_7 = min(in_warm_temp, in_high_humid, in_high_co2)

    consequence_1 = np.minimum(antecedent_1, off_hvac)
    consequence_2 = np.minimum(antecedent_2, low_hvac)
    consequence_3 = np.minimum(antecedent_3, low_hvac)
    consequence_4 = np.minimum(antecedent_4, medium_hvac)
    consequence_5 = np.minimum(antecedent_5, medium_hvac)
    consequence_6 = np.minimum(antecedent_6, high_hvac)
    consequence_7 = np.minimum(antecedent_7, high_hvac)

    # Aggregate
    return np.maximum.reduce(
        [
            consequence_1,
            consequence_2,
            consequence_3,
            consequence_4,
            consequence_5,
            consequence_6,
            consequence_7,
        ]
    )


def dominant_category(name, memberships):
    # memberships is a dict {category: value}
    dominant = max(memberships, key=memberships.get)
    print(f"{name} Category: {dominant} (μ={memberships[dominant]:.2f})")
    return dominant


def hvac_control_app(in_temp=None, in_humid=None, in_co2=None):
    print("HVAC Control System using Fuzzy Logic")

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
        "Warm": in_warm_temp,
    }
    humid_memberships = {
        "Dry": in_dry_humid,
        "Normal": in_normal_humid,
        "High": in_high_humid,
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
    r = evaluate_rules()

    # Defuzzification
    res = mf.defuzzify_trap(hvac, r)

    # HVAC category (compare crisp output to membership functions)
    hvac_memberships = {
        "Off":np.max(np.minimum(r, off_hvac)),
        "Low": np.max(np.minimum(r, low_hvac)),
        "Medium": np.max(np.minimum(r, medium_hvac)),
        "High": np.max(np.minimum(r, high_hvac)),
    }

    res_cat = dominant_category("HVAC", hvac_memberships)

    print(f"Input Summary:")
    print(f" - Temperature: {in_temp:.1f} °C ({temp_cat})")
    print(f" - Humidity: {in_humid:.1f} % ({humid_cat})")
    print(f" - CO₂: {in_co2:.1f} ppm ({co2_cat})")
    print(f"Recommended HVAC Level: {res:.2f} % ({res_cat})")

    plt.figure(0, figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(temp, cold_temp, label="Cold", color="skyblue")
    plt.plot(temp, comfortable_temp, label="Comfortable", color="green")
    plt.plot(temp, warm_temp, label="Warm", color="red")

    plt.scatter(
        [in_temp, in_temp, in_temp], [in_cold_temp, in_comfortable_temp, in_warm_temp]
    )
    plt.xlabel("Temperature")
    plt.title(f"Input Temperature Fuzzification: {in_temp:.1f} °C")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(humid, dry_humid, label="Dry", color="skyblue")
    plt.plot(humid, normal_humid, label="Normal", color="green")
    plt.plot(humid, high_humid, label="High", color="red")

    plt.scatter(
        [in_humid, in_humid, in_humid], [in_dry_humid, in_normal_humid, in_high_humid]
    )
    plt.xlabel("Humidity")
    plt.title(f"Input Humidity Fuzzification: {in_humid:.1f} %")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(co2, low_co2, label="Low", color="skyblue")
    plt.plot(co2, medium_co2, label="Medium", color="green")
    plt.plot(co2, high_co2, label="High", color="red")

    plt.scatter([in_co2, in_co2, in_co2], [in_low_co2, in_medium_co2, in_high_co2])
    plt.xlabel("CO2")
    plt.title(f"Input CO₂ Fuzzification: {in_co2:.1f} ppm")
    plt.legend()

    plt.figure(1, figsize=(12, 6))
    plt.plot(hvac, off_hvac, label="Off", color="skyblue")
    plt.plot(hvac, low_hvac, label="Low", color="green")
    plt.plot(hvac, medium_hvac, label="Medium", color="orange")
    plt.plot(hvac, high_hvac, label="High", color="red")
    plt.fill_between(hvac, np.zeros_like(hvac), r, color="orange", alpha=0.7)
    plt.scatter([res], [0], color="red", label="Defuzzified Output")
    plt.xlabel("HVAC Level")
    plt.title("Defuzzification using Centroid Method")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # # Sample input values
    in_temp = 21.5  # Current indoor temperature in °C
    in_humid = 42  # Current indoor humidity in %
    in_co2 = 600  # Current CO2 concentration in ppm
    hvac_control_app(in_temp, in_humid, in_co2)
