import numpy as np
from scipy.optimize import curve_fit


class Nonlinear_Regression():

    def fit_exp(self, xdata, ydata):
        popt, pcov = curve_fit(self.exponential_func, xdata, ydata, p0=[1, 0])
        return popt, pcov

    def exponential_func(self, x, a, b):
        return a * np.exp(b * x)

    def fit_log(self, xdata, ydata):
        popt, pcov = curve_fit(self.log_func, xdata, ydata, p0=[0, 1])
        return popt, pcov

    def log_func(self, x, a, b):
        return a * np.log(x) + b

    def fit_lipow(self, xdata, ydata):
        popt, pcov = curve_fit(self.lipow_func, xdata, ydata)
        return popt, pcov

    def lipow_func(self, x, a, b):
        return a - (b / x)

    def fit_gom(self, xdata, ydata):
        popt, pcov = curve_fit(self.gom_func, xdata, ydata, maxfev=200000, p0=[100, -4, (-1/400)*np.log((-1/4)*np.log(5/100))])
        return popt, pcov

    def gom_func(self, x, a, b, c):
        return a * np.exp(-b*np.exp(-c*x))

    def fit_pow(self, xdata, ydata):
        popt, pcov = curve_fit(self.pow_func, xdata, ydata, maxfev=200000, p0=[1, 1])
        return popt, pcov

    def pow_func(self, x, a, b):
        return a * np.power(x, b)