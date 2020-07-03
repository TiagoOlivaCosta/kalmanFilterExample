import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 20, 100)  # time
x_dot = 20*np.ones(len(t))  # m/s


def speed(x_dot):
    # Speedometer measurement model
    # consider speeds from 0 to 120 km/h
    sigma_nu = 0.01*120  # km/h
    sigma_nu = sigma_nu/3.6  # m/s
    nu = np.random.randn()*sigma_nu
    b = 1 + (np.random.random()-0.5)*0.1
    vel = b*x_dot + nu
    return vel


plt.plot(t, x_dot)
y = [speed(i) for i in x_dot]
plt.plot(t, y)
plt.show()
