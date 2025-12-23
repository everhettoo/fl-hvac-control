import numpy as np
import matplotlib.pyplot as plt
"""
Increasing (ramp) fuzzy membership function (also called "S-shaped" linear ramp for this simple form).

Definition:
  mu(x) = 0                      for x <= a
  mu(x) = (x - a) / (b - a)      for a < x < b
  mu(x) = 1                      for x >= b

Notes / Edge cases:
  - a must be strictly less than b. If a == b the function behaves like a step at a
    (0 for x <= a, 1 for x > a). If a > b a ValueError is raised because the
    interval is invalid.
  - The function is vectorized: it accepts numpy arrays and scalars.
"""
def inc(x, a, b):
    """
    Calculate the increasing (ramp) fuzzy membership function value.

    Parameters:
      x : scalar or numpy array
        Input value(s) where the membership is evaluated.
      a : float
        Left foot — the value where membership starts increasing from 0.
      b : float
        Right foot — the value where membership reaches 1.

    Returns:
      float or numpy.ndarray
        Membership degree(s) in the interval [0, 1].
    """
    if x<=a:
        return 0
    elif x>=b:
        return 1
    else:
        return (x-a)/(b-a)

# Quick demo when run as a script
if __name__ == "__main__":
    print("\nDemo of increasing_membership")
    inp = np.arange(0, 10)
    mu = [inc(x, 3, 5) for x in inp]

    plt.figure(figsize=(6, 3))
    plt.plot(inp, mu, '-o')
    plt.title('Increasing (ramp) membership function')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()