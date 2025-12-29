# This fuzzy membership functions from module L2-02112025-2
import numpy as np

"""
Increasing Membership Function.

return y:
    y=0, when x<=a
    y=1, when x>=b
    y=(x-a)/(b-a), when a<x<b    
"""


def inc(x, a, b):
    if x <= a:
        return 0
    elif x >= b:
        return 1
    else:
        return (x - a) / (b - a)


"""
Decreasing Membership Function.

return y:
    y=1, when x<=a
    y=0, when x>=b
    y=1-(x-a)/(b-a), when a<x<b
"""


def dec(x, a, b):
    return 1 - inc(x, a, b)


"""
Triangular Membership Function. Inlcudes incresing and decreasing internally.

return y:
    y=0, when x<=a
    y=(x-a)/(b-a), when a<x<=b
    y=(c-x)/(c-b), when b<x<=c
    y=0, when x>=c  
"""


def tri(x, a, b, c):
    if x <= a:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0


"""
Trapezoidal Membership Function. 

return y:
    y=0, when x<=a
    y=(x-a)/(b-a), when a<x<=b
    y=1, when b<x<=c
    y=(d-x)/(d-c), when c<x<=d
    y=0, when x>=d  
"""


def trap(x, a, b, c, d):
    if x <= c:
        return inc(x, a, b)
    else:
        return dec(x, c, d)


"""
Gaussian Membership Function. Gaussian function can change based on the requirements.
    m: mean (center)
    s: standard deviation (width)
"""


def gaussian(x, m, s):
    return np.exp(-((x - m) ** 2) / s**2)


"""
Sigmoid Membership Function. Always prefered over inc and dec.
    a: slope parameter (positive for increasing, negative for decreasing)
    b: center point (x value for y=0.5)
"""


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


"""
Defuzzification using the Centroid (Center of Gravity) method.
    Parameters:
        universe (array-like): Discrete universe of discourse (x-axis values).
        aggregated (array-like): Aggregated membership values (y-axis values).
    Returns:
        float: Crisp output value representing the defuzzified result.
"""


def defuzzify(universe, aggregated):
    if np.sum(aggregated) == 0:
        return 0.0
    return np.sum(universe * aggregated) / np.sum(aggregated)
