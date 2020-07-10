import unittest
import estimators
import numpy as np


class TestCovInt(unittest.TestCase):
    def test_perfcrit1(self):
        covint = estimators.CovarianceIntersection(
            performance_criterion='DeTerminaNt')
        self.assertEqual(covint.performance_criterion, np.linalg.det)

    def test_perfcrit2(self):
        with self.assertRaises(AssertionError):
            estimators.CovarianceIntersection(performance_criterion=2)

    def test_perfcrit3(self):
        with self.assertRaises(ValueError):
            estimators.CovarianceIntersection(performance_criterion='')


if __name__ == "__main__":
    # set up test process according to "An Elementary Introduction to Kalman
    # Filtering" - Pei, S.; Fussel, D.S.; Biswas, S.; Pingali, K. (2019)
    # Section 6 - Falling body example
    A = np.array([[1, 0], [0.25, 1]])
    B = np.array([[0, 0.25], [0, 0.5*0.25**2]])
    Q = np.array([[2, 2.5], [2.5, 4]])
    P0 = np.array([[80, 0], [0, 10]])
    x0 = np.zeros((2, 1))
    unittest.main()
