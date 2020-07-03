import numpy as np
import process


class Kalman:
    def __init__(self, model, process_cov, noise_cov):
        self.F, self.B, self.H = model.A, model.B, model.C
        self.Q, self.R = process_cov, noise_cov
        assert self.F.shape == self.Q.shape
        assert self.H.shape == self.Q.shape

        # initialize state and covariance estimates
        self.x = np.zeros((self.F.shape[0], 1))
        self.P = np.zeros((self.F.shape[0], self.F.shape[0]))

    def prediction(self, system_input):
        x, pp, u = self.x, self.P, system_input
        x_pri = self.F.dot(x) + self.B.dot(u)
        pp_pri = (self.F.dot(pp)).dot(self.F.T) + self.Q
        return x_pri, pp_pri

    def update(self, x_pri, pp_pri, measurement):
        x, pp, z = self.x, self.P, measurement
        y_s = z - self.H.dot(x)  # inovation, pre-fit residue
        ss = (self.H.dot(pp)).dot(self.H.T) + self.R  # inovation covariance
        kk = self.pp.dot(self.H.T).dot(ss.I)  # Kalman gain
        x_pos = x_pri + kk.dot(y_s)
        pp_pos = (np.eye(pp.shape) - np.dot(kk, self.H)).dot(pp_pri)
        self.x, self.P = x_pos, pp_pos

    def filter(self, system_input, measurement):
        x_pri, pp_pri = self.prediction(system_input)
        self.update(x_pri, pp_pri, measurement)


if __name__ == "__main__":
    kf = Kalman(process.sys, process.Q)
