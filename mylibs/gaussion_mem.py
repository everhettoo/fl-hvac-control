import matplotlib.pyplot as plt
"""
Gaussian (normal) membership function centered at `mean` with standard deviation `sigma`.

Definition:
  mu(x) = exp(-((x - m) / s)^2)

where:
  - m = mean (center) — where mu(m) = 1 (peak of the bell curve)
  - s = sigma (standard deviation) — controls the width/spread

Notes / Edge cases:
  - sigma must be positive, otherwise a ValueError is raised.
  - At x = m: mu(m) = exp(0) = 1.0 (maximum membership)
  - At x = m ± s: mu(m±s) = exp(-1) ≈ 0.3679 (one standard deviation)
  - At x = m ± 2s: mu(m±2s) = exp(-4) ≈ 0.0183 (two standard deviations)
  - The function is vectorized and returns a scalar when given a scalar input.
  - The curve is symmetric around the mean m.
"""
import numpy as np

def gauss(x, m, s):
    """
    Calculate the Gaussian fuzzy membership function value.

    Parameters:
      x : scalar or numpy array
        Input value(s) to evaluate the membership.
      m : float
        The center (peak) of the Gaussian where mu == 1.
      s : float
        Standard deviation (spread). Must be > 0.

    Returns:
      numpy.ndarray or float (same shape as x)
        Membership degrees in (0, 1], with mu(mean) == 1.
    """
    return np.exp(-((x - m) ** 2) / s**2)

if __name__ == "__main__":
    print("\nDemo of gaussion_membership")
    inp = np.arange(0, 10, 0.1)
    mu = [gauss(x, 5, 3) for x in inp]

    plt.figure(figsize=(6, 3))
    plt.plot(inp, mu)
    plt.title('Gaussion (ramp) membership function')
    plt.xlabel('x')
    plt.ylabel('membership')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()