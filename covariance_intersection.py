from scipy.optimize import fminbound
from numpy.linalg import inv, det
import numpy as np


class CovarianceIntersection:
    """
    """
    def __init__(self, performance_criterion="Trace"):
        assert type(performance_criterion) is str
        performance_criterion = performance_criterion.lower()
        if performance_criterion == "trace":
            self.performance_criterion = np.trace
        elif performance_criterion == "determinant":
            self.performance_criterion = det
        else:
            raise ValueError
        self.algorithm_name = "Covariance Intersection"
        self.algorithm_abbreviation = "CI"

        self.omega = 0
        self.state = np.array()

    @staticmethod
    def cov_weighting(cov_a, cov_b, omega):
        info_a, info_b = inv(cov_a), inv(cov_b)
        w_a, w_b = np.multiply(omega, info_a), np.multiply(1 - omega, info_b)
        return w_a, w_b

    def fuse_cov(self, cov_a, cov_b, omega):
        w_info_a, w_info_b = self.cov_weighting(cov_a, cov_b, omega)
        info_c = inv(w_info_a + w_info_b)
        return info_c

    def fuse(self, mean_a, cov_a, mean_b, cov_b):
        omega = self.optimize_omega(cov_a, cov_b)
        w_a, w_b = self.cov_weighting(cov_a, cov_b, omega)
        cov = inv(w_a + w_b)
        mean = np.dot(cov, (np.dot(w_a, mean_a) + np.dot(w_b, mean_b)))
        return mean, cov

    def fusion(self, estimate_a, estimate_b):
        assert type(estimate_a) is tuple and type(estimate_b) is tuple
        x_a, cov_a = estimate_a
        x_b, cov_b = estimate_b


    def optimize_omega(self, cov_a, cov_b):
        def optimize_fn(omega):
            return self.performance_criterion(self.fuse_cov(cov_a, cov_b,
                                                            omega))
        return fminbound(optimize_fn, 0, 1)
        #  return 0.5
