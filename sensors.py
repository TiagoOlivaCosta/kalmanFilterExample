import numpy as np
import matplotlib.pyplot as plt


class Speedometer:
    def __init__(self, mode="biased"):
        assert type(mode) is str
        self.mode = mode.lower()
        if self.mode == "biased":
            self.bias = 1 + (np.random.random()-0.5)*0.1
            print("Speedometer bias: ", self.bias)

    def measurement(self, x_dot):
        # Speedometer measurement model
        # consider speeds from 0 to 120 km/h
        sigma_nu = 0.05*120  # km/h
        sigma_nu = sigma_nu/3.6  # m/s
        nu = np.random.randn()*sigma_nu
        if self.mode == "biased":
            return self.bias*x_dot + nu
        elif self.mode == "unbiased":
            return x_dot + nu


class GPS:
    def __init__(self):
        pass

    def measurement(self, x):
        # "autonomous civilian GPS horizontal position fixes
        #  are typically accurate to about 15 meters "  - Wikipedia
        # entao 2*sigma = 15m
        sigma_nu = 15.0/2  # m
        nu = np.random.randn()*sigma_nu
        return x + nu


def msq_error(x, y):
    return sum((x-y)**2)/len(x)


if __name__ == "__main__":
    t = np.linspace(0, 20, 1000)  # time
    x_dot = 20*np.ones(len(t))  # m/s
    x = t*x_dot  # position (m)

    # speed
    plt.plot(t, x_dot)
    speedometer = Speedometer()
    y_dot = [speedometer.measurement(i) for i in x_dot]
    plt.plot(t, y_dot)
    msqerror = np.sqrt(msq_error(x_dot, y_dot))
    avgerror = np.average(y_dot-x_dot)
    print(f"Sqrt of MSq speed error: {msqerror:.2f} m/s")
    print(f"Average error: {avgerror:.2f} m/s")
    plt.show()

    # speed biased vs unbiased 
    plt.plot(t, x_dot)
    speedometer_unbiased = Speedometer(mode="unbiased")
    y_dot = [speedometer.measurement(i) for i in x_dot]
    y_dot_unbiased = [speedometer_unbiased.measurement(i) for i in x_dot]
    plt.plot(t, y_dot)
    plt.plot(t, y_dot_unbiased)
    msqerror = np.sqrt(msq_error(x_dot, y_dot_unbiased))
    avgerror = np.average(y_dot_unbiased-x_dot)
    print(f"Sqrt of MSq speed error (unbiased): {msqerror:.2f} m/s")
    print(f"Average error (unbiased): {avgerror:.2f} m/s")
    plt.show()

    # position
    gps = GPS()
    y = [gps.measurement(i) for i in x]
    plt.plot(t, x)
    plt.plot(t, y)
    plt.show()
    msqerror = np.sqrt(msq_error(x, y))
    avgerror = np.average(y-x)
    print(f"Sqrt of MSq  GPS error: {msqerror:.2f} m")
    print(f"Average GPS error: {avgerror:.2f} m")
