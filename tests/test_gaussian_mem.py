import unittest
import numpy as np
import mylibs.membership_functions as mf

class TestGaussianMembershipFunction(unittest.TestCase):

    def test_returns_one_at_mean(self):
        self.assertEqual(mf.gaussian(10, 10, 1), 1)
        self.assertEqual(mf.gaussian(0, 0, 1), 1)
        self.assertEqual(mf.gaussian(100, 100, 0.5), 1)

    def test_returns_less_than_one_away_from_mean(self):
        result = mf.gaussian(11, 10, 1)
        self.assertLess(result, 1)
        self.assertGreater(result, 0)

    def test_returns_symmetric_values_around_mean(self):
        m, s = 10, 2
        left = mf.gaussian(8, m, s)
        right = mf.gaussian(12, m, s)
        self.assertAlmostEqual(left, right)

    def test_returns_decreasing_values_with_distance_from_mean(self):
        m, s = 10, 1
        at_mean = mf.gaussian(m, m, s)
        one_sigma_away = mf.gaussian(m + s, m, s)
        two_sigma_away = mf.gaussian(m + 2*s, m, s)

        self.assertGreater(at_mean, one_sigma_away)
        self.assertGreater(one_sigma_away, two_sigma_away)

    def test_returns_approximately_zero_point_three_at_one_sigma(self):
        m, s = 10, 1
        result = mf.gaussian(m + s, m, s)
        self.assertAlmostEqual(result, np.exp(-1), places=5)

    def test_returns_approximately_zero_point_zero_one_at_three_sigma(self):
        m, s = 10, 1
        result = mf.gaussian(m + 3*s, m, s)
        self.assertAlmostEqual(result, np.exp(-9), places=5)

    def test_handles_negative_values(self):
        result = mf.gaussian(-5, 0, 1)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)
        self.assertAlmostEqual(result, mf.gaussian(5, 0, 1))

    def test_handles_negative_mean(self):
        m, s = -10, 2
        self.assertEqual(mf.gaussian(m, m, s), 1)
        self.assertLess(mf.gaussian(m + 1, m, s), 1)

    def test_handles_decimal_values(self):
        result = mf.gaussian(2.5, 2.5, 0.5)
        self.assertEqual(result, 1)

    def test_handles_very_small_sigma(self):
        m, s = 10, 0.01
        result_at_mean = mf.gaussian(m, m, s)
        result_far = mf.gaussian(m + 1, m, s)

        self.assertEqual(result_at_mean, 1)
        self.assertLess(result_far, 1e-300)

    def test_handles_very_large_sigma(self):
        m, s = 10, 100
        result_at_mean = mf.gaussian(m, m, s)
        result_far = mf.gaussian(m + 50, m, s)

        self.assertEqual(result_at_mean, 1)
        self.assertGreater(result_far, 0.7)

    def test_handles_x_as_numpy_array(self):
        x_array = np.array([8, 9, 10, 11, 12])
        result = mf.gaussian(x_array, 10, 1)

        self.assertEqual(len(result), 5)
        self.assertEqual(result[2], 1)
        self.assertAlmostEqual(result[1], result[3])
        self.assertAlmostEqual(result[0], result[4])

    def test_returns_scalar_for_scalar_input(self):
        result = mf.gaussian(15, 10, 1)
        self.assertIsInstance(result, (int, float, np.floating))

    def test_returns_array_for_array_input(self):
        result = mf.gaussian(np.array([10, 11, 12]), 10, 1)
        self.assertIsInstance(result, np.ndarray)

    def test_handles_zero_centered_gaussian(self):
        m, s = 0, 1
        self.assertEqual(mf.gaussian(0, m, s), 1)
        self.assertAlmostEqual(mf.gaussian(-1, m, s), mf.gaussian(1, m, s))

    def test_produces_valid_membership_values(self):
        m, s = 10, 2
        test_points = np.linspace(0, 20, 100)
        results = mf.gaussian(test_points, m, s)

        for result in results:
            self.assertGreater(result, 0)
            self.assertLessEqual(result, 1)

    def test_handles_consecutive_calls_with_same_parameters(self):
        result1 = mf.gaussian(15, 10, 1)
        result2 = mf.gaussian(15, 10, 1)
        self.assertEqual(result1, result2)

    def test_handles_co2_sensor_range(self):
        m, s = 800, 200
        self.assertEqual(mf.gaussian(m, m, s), 1)
        self.assertAlmostEqual(mf.gaussian(m + s, m, s), np.exp(-1))
        self.assertLess(mf.gaussian(1500, m, s), mf.gaussian(800, m, s))

    def test_handles_temperature_sensor_range(self):
        m, s = 23, 2
        self.assertEqual(mf.gaussian(m, m, s), 1)
        self.assertAlmostEqual(mf.gaussian(m + s, m, s), np.exp(-1))
        self.assertGreater(mf.gaussian(m - 1, m, s), 0.7)

    def test_handles_humidity_sensor_range(self):
        m, s = 55, 10
        self.assertEqual(mf.gaussian(m, m, s), 1)
        self.assertAlmostEqual(mf.gaussian(m + s, m, s), np.exp(-1))
        self.assertGreater(mf.gaussian(m - 5, m, s), 0.7)

    def test_bell_curve_is_smooth_and_continuous(self):
        m, s = 10, 1
        x_values = np.linspace(m - 5*s, m + 5*s, 200)
        y_values = mf.gaussian(x_values, m, s)

        for i in range(len(y_values) - 1):
            diff = abs(y_values[i] - y_values[i + 1])
            self.assertLess(diff, 0.05)

    def test_peak_is_at_mean(self):
        m, s = 10, 2
        left = mf.gaussian(m - 0.1, m, s)
        peak = mf.gaussian(m, m, s)
        right = mf.gaussian(m + 0.1, m, s)

        self.assertLess(left, peak)
        self.assertLess(right, peak)
        self.assertEqual(peak, 1)

    def test_wider_sigma_produces_wider_curve(self):
        m = 10
        x = 12
        narrow = mf.gaussian(x, m, 0.5)
        wide = mf.gaussian(x, m, 2)

        self.assertLess(narrow, wide)

    def test_narrower_sigma_produces_narrower_curve(self):
        m = 10
        x = 12
        narrow = mf.gaussian(x, m, 0.5)
        wide = mf.gaussian(x, m, 2)

        self.assertLess(narrow, wide)

    def test_handles_very_large_distance_from_mean(self):
        result = mf.gaussian(1000, 0, 1)
        self.assertEqual(result, 0)

    def test_handles_large_distance_produces_small_values(self):
        m, s = 10, 1
        result = mf.gaussian(m + 10*s, m, s)
        self.assertLess(result, 1e-20)

    def test_output_approaches_zero_but_never_reaches(self):
        m, s = 10, 1
        result = mf.gaussian(m + 50*s, m, s)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 1e-300)

    def test_multivariate_input_preserves_symmetry(self):
        m, s = 50, 5
        x_values = np.array([40, 45, 55, 60])
        results = mf.gaussian(x_values, m, s)

        self.assertAlmostEqual(results[0], results[3])
        self.assertAlmostEqual(results[1], results[2])

    def test_consistent_with_normal_distribution_properties(self):
        m, s = 0, 1
        at_mean = mf.gaussian(m, m, s)
        at_one_sigma = mf.gaussian(m + s, m, s)
        at_two_sigma = mf.gaussian(m + 2*s, m, s)

        self.assertEqual(at_mean, 1)
        self.assertAlmostEqual(at_one_sigma, np.exp(-1))
        self.assertAlmostEqual(at_two_sigma, np.exp(-4))


if __name__ == "__main__":
    unittest.main()

