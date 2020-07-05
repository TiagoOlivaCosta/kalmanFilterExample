import numpy as np
import matplotlib.pyplot as plt
import control
import covariance_intersection as CI
import process
import sensors


class Kalman:
    def __init__(self, model, process_cov, noise_cov):
        self.F, self.B, self.H = model.A, model.B, model.C
        self.Q, self.R = process_cov, noise_cov
        assert self.F.shape == self.Q.shape
        assert self.H.dot(self.H.T).shape == self.R.shape

        # initialize state and covariance estimates
        self.x = np.zeros((self.F.shape[0], 1))
        self.P = np.zeros((self.F.shape[0], self.F.shape[0]))

    def prediction(self, system_input):
        x, pp, u = self.x, self.P, system_input
        x_pri = self.F.dot(x) + self.B.dot(u)
        pp_pri = (self.F.dot(pp)).dot(self.F.T) + self.Q
        return x_pri, pp_pri

    def update(self, x_pri, pp_pri, measurement):
        x, pp, z = self.x, pp_pri, measurement
        y_s = z - self.H.dot(x)  # inovation, pre-fit residue
        ss = (self.H.dot(pp)).dot(self.H.T) + self.R  # inovation covariance
        kk = pp.dot(self.H.T).dot(ss.I)  # Kalman gain
        x_pos = x_pri + kk.dot(y_s)
        pp_pos = (np.eye(len(pp)) - np.dot(kk, self.H)).dot(pp)
        self.x, self.P = x_pos, pp_pos

    def filter(self, system_input, measurement):
        x_pri, pp_pri = self.prediction(system_input)
        self.update(x_pri, pp_pri, measurement)


if __name__ == "__main__":
    ground_truth = process.sys
    dt = process.T  # sampling time (s)
    time = np.arange(0, 21, dt)
    _, x = control.step_response(ground_truth, time)
    plt.plot(time, x[0])

    # speedometer
    h_speed = np.array([[0, 1]])  # observation matrix
    var_speedometer = 1.7**2
    r_speed = np.array([[var_speedometer]])
    sys_speed = control.ss(process.A, process.B, h_speed, 0, dt)
    kf_speed = Kalman(sys_speed, process.Q, r_speed)
    kf_speed_unbiased = Kalman(sys_speed, process.Q, r_speed)
    speedometer = sensors.Speedometer()
    speedometer_unbiased = sensors.Speedometer(mode="unbiased")
    x_speedometer = np.zeros(len(time))
    z_speedometer = np.zeros(len(time))
    x_speedometer_unbiased = np.zeros(len(time))
    z_speedometer_unbiased = np.zeros(len(time))
    for i, t in enumerate(time):
        z = speedometer.measurement(x[1][i])
        z_speedometer[i] = z
        kf_speed.filter(0, z)
        x_speedometer[i] = kf_speed.x[0]

        # unbiased version
        z = speedometer_unbiased.measurement(x[1][i])
        z_speedometer_unbiased[i] = z
        kf_speed_unbiased.filter(0, z)
        x_speedometer_unbiased[i] = kf_speed_unbiased.x[0]
    plt.plot(time, x_speedometer)
    plt.plot(time, x_speedometer_unbiased)
    plt.show()

    # GPS
    h_gps = np.array([[1, 0]])  # observation matrix
    var_gps = 7**2
    r_gps = np.array([[var_gps]])
    sys_gps = control.ss(process.A, process.B, h_gps, 0, dt)
    kf_gps = Kalman(sys_gps, process.Q, r_gps)
    gps = sensors.GPS()
    x_gps = np.zeros(len(time))
    z_gps = np.zeros(len(time))
    for i, t in enumerate(time):
        z = gps.measurement(x[0][i])
        z_gps[i] = z
        kf_gps.filter(0, z)
        x_gps[i] = kf_gps.x[0]
    plt.plot(time, x_gps)
    plt.plot(time, z_gps)
    plt.show()

    plt.plot(time, x[0])
    plt.plot(time, x_speedometer)
    plt.plot(time, x_gps)

    # Kalman fusion biased
    H = np.eye(2)
    sys_fused = control.ss(process.A, process.B, H, 0, dt)
    r_fused = np.diag([var_gps, var_speedometer])
    kf_fused = Kalman(sys_fused, process.Q, r_fused)
    gps = sensors.GPS()
    x_fused = np.zeros(len(time))
    z_fused = np.zeros((len(time), 2))
    for i, t in enumerate(time):
        z = np.array(gps.measurement(x[0][i]),
                     speedometer.measurement(x[1][i]))
        z_fused[i, :] = z
        kf_fused.filter(0, z)
        x_fused[i] = kf_fused.x[0]
    plt.plot(time, x_fused)
    plt.plot(time, x_gps)
    plt.plot(time, z_gps)
    # plt.plot(time, x_speedometer)
    plt.plot(time, x[0])
    plt.show()


    # Kalman fusion unbiased
    kf_unbiased = Kalman(sys_fused, process.Q, r_fused)
    x_unbiased = np.zeros(len(time))
    z_unbiased = np.zeros((len(time), 2))
    for i, t in enumerate(time):
        z = np.array(gps.measurement(x[0][i]),
                     speedometer_unbiased.measurement(x[1][i]))
        z_unbiased[i, :] = z
        kf_unbiased.filter(0, z)
        x_unbiased[i] = kf_unbiased.x[0]


    # CI fusion
    kf_gps = Kalman(sys_gps, process.Q, r_gps)
    kf_speed = Kalman(sys_speed, process.Q, r_speed)
    covint = CI.CovarianceIntersection()
    x_ci = np.zeros(len(time))
    for i, t in enumerate(time):
        z_g, z_s = gps.measurement(x[0][i]), speedometer.measurement(x[1][i])
        kf_gps.filter(0, z_g), kf_speed.filter(0, z_s)
        ci, _ = covint.fuse(kf_gps.x, kf_gps.P, kf_speed.x, kf_speed.P)
        x_ci[i] = ci[0]
    plt.plot(time, x[0], label='model')
    plt.plot(time, x_ci, label='covariance intersection')
    plt.plot(time, x_fused, label='kalman')
    plt.legend(loc='lower right')
    plt.show()
