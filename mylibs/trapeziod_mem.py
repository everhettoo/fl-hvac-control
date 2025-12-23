import numpy as np
import matplotlib.pyplot as plt
"""
Trapezoidal membership function.

Definition:
  mu(x) = 0                      for x <= a
  mu(x) = (x - a) / (b - a)      for a < x < b
  mu(x) = 1                      for b <= x <= c
  mu(x) = (d - x) / (d - c)      for c < x < d
  mu(x) = 0                      for x >= d

Edge cases:
  - If a > b or b > c or c > d -> ValueError.
  - If a == b then left slope is a step (mu(a)==1).
  - If c == d then right slope is a step (mu(d)==1).
  - If b == c then trapezoid collapses to a triangle with peak at [b,c].
  - If a == b == c == d -> singleton: mu==1 at x==a else 0.

The function is vectorized and returns a scalar float for scalar input.
"""
from mylibs.increasing_mem import inc
from mylibs.decreasing_mem import dec

def trap(x, a, b, c, d):
    """
    Calculate the trapezoidal fuzzy membership function value.

    Parameters:
      x : scalar or numpy array
      a : float
        Left foot — the value where membership starts increasing from 0.
      b : float
        Right foot of left slope — the value where membership reaches 1.
      c : float
        Left foot of right slope — the value where membership starts decreasing from 1.
      d : float
        Right foot — the value where membership reaches 0.
      a, b, c, d : numbers with a <= b <= c <= d
        The trapezoid points. b..c is the plateau where mu==1.

    Returns:
      numpy.ndarray or float (same shape as x)
        Membership values in [0,1].
    """
    if x<=c :
        return inc(x, a, b)
    else:
        return dec(x, c, d)

if __name__ == "__main__":
    print("\nDemo of trapezoidal_membership")
    inp = np.arange(0, 10)
    mu = [trap(x, 3, 4, 5, 7) for x in inp]

    plt.figure(figsize=(6, 3))
    plt.plot(inp, mu)
    plt.title('Trapezoidal (ramp) membership function')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()