import numpy as np
import matplotlib.pyplot as plt
"""
Triangular membership function.

Definition (standard):
  mu(x) = 0                   for x <= a or x >= c
  mu(x) = (x - a) / (b - a)   for a < x < b
  mu(x) = (c - x) / (c - b)   for b <= x < c
  mu(b) = 1

Parameters:
  x : scalar or numpy array
    Input value(s) where the membership is evaluated.
  a, b, c : floats
    Triangle points with a <= b <= c. 'b' is the peak (mu=1).

Returns:
  numpy.ndarray or float (same shape as x)
    Membership degree(s) in [0,1].

Edge cases:
  - If a > b or b > c -> ValueError.
  - If a == b == c -> returns 1.0 at x==a, else 0.0 (a crisp singleton).
  - If a == b < c -> left slope is a step: mu(a)==1, then falls to 0 at c.
  - If a < b == c -> right slope is a step: mu(c)==1, left rises to 1 at b.

The function is fully vectorized and returns a scalar when given a scalar.
"""
def tri(x, a, b, c):
    """
    Calculate the triangular fuzzy membership function value.

    Parameters:
      x : scalar or numpy array
        Input value(s) where the membership is evaluated.
      a, b, c : floats
        Triangle points with a <= b <= c. 'b' is the peak (mu=1).

    Returns:
      numpy.ndarray or float (same shape as x)
        Membership degree(s) in [0,1].
    """
    if x <= a:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0

if __name__ == "__main__":
    print("\nDemo of triangular_membership")
    inp = np.arange(0, 10)
    mu = [tri(x, 3, 5, 8) for x in inp]

    plt.figure(figsize=(6, 3))
    plt.plot(inp, mu)
    plt.title('Triangular (ramp) membership function')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()