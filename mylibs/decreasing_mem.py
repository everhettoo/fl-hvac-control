import numpy as np
import matplotlib.pyplot as plt
"""
Decreasing (falling/ramp) fuzzy membership function.

Definition:
  mu(x) = 1                      for x <= a
  mu(x) = (b - x) / (b - a)      for a < x < b
  mu(x) = 0                      for x >= b

Notes / Edge cases:
  - a must be <= b. If a == b the function behaves like a step at a
    (1 for x < a, 0 for x >= a). If a > b a ValueError is raised.
  - The function is vectorized: it accepts numpy arrays and scalars.
"""
from mylibs.increasing_mem import inc

def dec(x, a, b):
    """
    Calculate the decreasing (ramp) fuzzy membership function value.

    Parameters:
      x : scalar or numpy array
        Input value(s) where the membership is evaluated.
      a : float
        Left foot — the value where membership starts decreasing from 1.
      b : float
        Right foot — the value where membership reaches 0.

    Returns:
      numpy.ndarray or float (same shape as x)
        Membership degree(s) in the interval [0, 1].
    """
    return 1 - inc(x, a, b)

# Quick demo when run as a script
if __name__ == "__main__":
    print("\nDemo of decreasing_membership")
    inp = np.arange(0, 10)
    mu = [dec(x, 3, 5) for x in inp]

    plt.figure(figsize=(6, 3))
    plt.plot(inp, mu, '-s', label='decreasing')
    plt.title('Decreasing (ramp) membership functions')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()
