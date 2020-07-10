from abc import ABC
import abc
from scipy.optimize import fminbound
from numpy.linalg import inv, det
import numpy as np


# ____________________ SETUP: abstract classes ____________________

class Estimate:
    def __init__(self, dimension=2):
        dim = dimension
        self.state = np.zeros((dim, 1))
        self.covariance = np.zeros((dim, dim))


class Estimator(ABC):
    def __init__(self, dimension=2):
        # self.estimate = Estimate()
        self.state = np.zeros((dimension, 1))
        self.covariance = np.zeros(dimension)

    @abc.abstractmethod
    def update(self):
        pass


class Fusor(ABC):
    def __init__(self):
        self._x = Estimate().state
        self._P = Estimate().covariance
        self._w = 0.5  # fusion gain, starts at naive fusion

    @abc.abstractmethod
    def fuse(self):
        pass


# ____________________ Covariance Based ____________________

class CovarianceIntersection(Fusor):
    def __init__(self, performance_criterion="Trace"):
        super().__init__()
        self.performance_criterion = performance_criterion
        self.algorithm_name = "Covariance Intersection"

    @property
    def performance_criterion(self):
        return self._perf_crit

    @performance_criterion.setter
    def performance_criterion(self, performance_criterion):
        assert type(performance_criterion) is str
        if performance_criterion.lower() == "trace":
            self._perf_crit = np.trace
        elif performance_criterion.lower() == "determinant":
            self._perf_crit = det
        else:
            raise ValueError

    @staticmethod
    def cov_fusion(cov_a, cov_b, omega):
        info_c = omega*inv(cov_a) + (1-omega)*inv(cov_b)
        cov_c = inv(info_c)
        return cov_c

    def omega(self, cov_a, cov_b):
        def optimize_fn(omega):
            cov_c = self.cov_fusion(cov_a, cov_b, omega)
            return self.performance_criterion(cov_c)
        return fminbound(optimize_fn, 0, 1)

    @property
    def fusion(self):
        return self._x, self._P, self._w

    @fusion.setter
    def fuse(self, estimate_a, estimate_b):
        x_a, cov_a = estimate_a
        x_b, cov_b = estimate_b
        w = self.omega(cov_a, cov_b)
        self._P = self.cov_fusion(cov_a, cov_b, w)
        self._x = self.P*(w*inv(cov_a)*x_a + (1-w)*inv(cov_b)*x_b)
        self._w = w


class InverseCovarianceIntersection(Fusor):
    pass


class EllipsoidalIntersection(Fusor):
    pass


# ____________________ Kalman Based ____________________

class KalmanFilter(Estimator):
    def __init__(self, model):
        self.A = model.A
        self.B = model.B


class UnscentedKalmanFilter(Estimator):
    pass


class ExtendedKalmanFilter(Estimator):
    pass
