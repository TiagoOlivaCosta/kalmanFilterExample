import numpy as np
import matplotlib.pyplot as plt
import control


T = 1  # sampling time
sigma_w = 1  # process variance (m/s)^2
# State transition matrix, F
F = np.array([[1, T],
              [0, 1]])
B = np.array([[0], [1]])  # input matrix
# Process Covariance matrix, Q
Q = np.array([[1./3*T**3, 0.5*T**2],
              [0.5*T**2, T]])*sigma_w**2
C = np.array([[1, 0]])  # observation matrix
sys = control.ss(F, B, C, 0, T)

if __name__ == "__main__":
    time = np.arange(0, 21, T)
    _, x = control.step_response(sys, time)
    plt.plot(time, x)
    plt.show()
