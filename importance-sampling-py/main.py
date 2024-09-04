import numpy as np
import matplotlib.pyplot as pyplot


class MertonJumpDiffusionModel:
    def __init__(self, s0, r, lambda_, sigma, a, b, t, steps) -> None:
        self.s0 = s0  # initial stock price
        self.r = r  # risk-free rate
        self.lambda_ = lambda_  # poisson jump intensity
        self.sigma = sigma  # volatility
        self.a = a  # mean jump value
        self.b = b  # standard deviation of jump
        self.t = t  # time horizon
        self.steps = steps  # time steps
        self.dt = t / steps  # delta t, or t_k+1 - t_k

    def simulate_returns_path(self):
        pass

    def simulate_price_path(self):
        pass

    def european_call_payoff(self):
        pass

    def european_put_payoff(self):
        pass


def plot_paths(model, num_paths=10):
    """
    TODO:
    Use matplotlib to plot the movement of the underlying with a default of 10 simulations
    """
    pass


def plot_option_price_convergence(model, strike, num_simulations=10000):
    """
    TODO:
    Use matplotlib to plot the convergence of the option price with 95% confidence intervals
    """
    pass


def main():
    print("Merton Jump Diffusion Model Simulation")


if __name__ == "__main__":
    main()
