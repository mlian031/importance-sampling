import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint


plt.style.use("ggplot")

"""
Objective:

Simulating the JDM at fixed dates from Glasserman (2003)
"""


def JDM():

    # given parameters

    t = 1  # time to maturity
    time_steps = 252  # daily
    lambda_ = 10  # jump intensity (frequency)
    a = 5  # mean jump value (also mu_j)
    b = 4  # standard deviation of the jump (also sigma_j)
    dt = t / time_steps  # delta time
    s0 = 14.58  # initial price
    mu = 0.18  # drift
    sigma = 0.22  # volatility
    r = 0.18  # risk free rate

    print("===================")
    print("Parameters")
    print("===================")

    print(f"time to maturity: {t}")
    print(f"time steps: {time_steps}")
    print(f"jump intensity (lambda): {lambda_}")
    print(f"mean jump value (a): {a}")
    print(f"jump standard deviation (b): {b}")
    print(f"delta time (dt): {dt:.4f}")
    print(f"initial price (s0): {s0}")
    print(f"drift (mu): {mu}")
    print(f"volatility (sigma): {sigma}")
    print(f"risk free rate (r): {r}")

    # generate Z ~ N(0,1)
    z = np.random.normal(0, 1)

    # generate N ~ Poisson(lambda * (tau_k+1 - tau_k))
    n = np.random.poisson(
        lam=(lambda_ * dt), size=time_steps
    )  # this is a poisson random variable

    # Note: The following code is useless
    # if n.any() > 0:
    # generate log Y_1, ... log Y_N, this is a jump random variable
    # log_y = np.random.normal(loc=a, scale=b, size=n)
    # m = np.sum(log_y)
    # else:
    # m = 0

    # I realize if the poisson random variable is 0,
    # then simply multiplying the jump random variable (y)
    # with the poisson random variable (n) will result in
    # the jump component in the log returns.

    jump_component = n * np.random.normal(loc=a, scale=b, size=n)

    times = np.linspace(0, t, time_steps + 1)

    log_returns = 



    return times, log_returns
