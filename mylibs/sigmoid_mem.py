import matplotlib.pyplot as plt
"""
Sigmoid (S-shaped) membership function. Always preferred over inc and dec.

Definition:
  mu(x) = 1 / (1 + exp(-a * (x - b)))

where:
  - 'b' is the midpoint (inflection point) where mu(b) == 0.5
  - 'a' controls the steepness (slope). Positive a -> increasing S-curve.
                                        Negative a -> decreasing S-curve.

Notes / Edge cases:
  - a must be non-zero, otherwise ValueError is raised.
  - At x = b, mu(b) == 0.5 exactly.
  - Large |a| -> steeper transition; small |a| -> smoother transition.
  - The function is vectorized and returns a scalar when given a scalar.
  - For a > 0: curve is increasing (like an S from bottom-left to top-right).
  - For a < 0: curve is decreasing (reflected/inverted S).

Example:
  x = np.linspace(-5, 5, 21)
  mu = sigmoid(x, a=1.0, b=0.0)  # increasing sigmoid centered at 0
"""
import numpy as np

def sigmoid(x, a, b):
    """
    Calculate the sigmoid fuzzy membership function value.

    Parameters:
      x : scalar or numpy array
        Input value(s) to evaluate the membership.
      a : float
        Steepness/slope coefficient (non-zero).
        a > 0 -> increasing S-curve; a < 0 -> decreasing S-curve.
      b : float
        Midpoint/center of the sigmoid curve (mu(b) == 0.5).

    Returns:
      numpy.ndarray or float (same shape as x)
        Membership degrees in (0, 1), approaching but never exactly 0 or 1.
    """
    return 1 / (1 + np.exp(-a*(x-b)))

if __name__ == "__main__":
    print("\nDemo of sigmoid_membership")
    inp = np.arange(0, 10, 0.1)
    mu = [sigmoid(x, 5, 3) for x in inp]

    plt.figure(figsize=(6, 3))
    plt.plot(inp, mu)
    plt.title('Sigmoid (ramp) membership function')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()