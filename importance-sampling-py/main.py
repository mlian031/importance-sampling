import numpy as np
import matplotlib.pyplot as pyplot


class MertonJumpDiffusionModel:
    def __init__(self, s0, r, mu, lambda_, sigma, a, b, t, steps) -> None:
        self.s0 = s0  # initial stock price
        self.r = r  # risk-free rate
        self.mu = mu  # drift
        self.lambda_ = lambda_  # poisson jump intensity
        self.sigma = sigma  # volatility
        self.a = a  # mean jump value
        self.b = b  # standard deviation of jump
        self.t = t  # time horizon
        self.steps = steps  # time steps
        self.dt = t / steps  # delta t, or t_k+1 - t_k

    def simulate_returns_path(self):
        # generate standard random variable
        z = np.random.normal(0, 1)
        # generate poisson random variable
        n = np.random.poisson(lam=(self.lambda_ * self.dt), size=self.steps)
        # generate jump component
        y = np.random.normal(loc=self.a, scale=self.b, size=self.steps)
        # get an array of n='steps' evenly spaced times
        t = np.linspace(
            0, self.t, self.steps + 1
        )  # splits it into n='steps' number of intervals

        log_returns = (
            self.s0
            * np.exp(self.dt * (self.mu - 0.5 * self.sigma**2))
            * np.exp(self.sigma * np.sqrt(self.dt) * z)
            * np.exp(n * y)
        )

        return times, log_returns

    def simulate_price_path(self):
        times, log_returns = self.simulate_returns_path()
        log_prices = np.cumsum(log_returns)
        prices = self.s0 * np.exp(np.insert(log_prices, 0, 0))

        return times, prices

    def european_call_payoff(self, strike):
        _, prices = self.simulate_price_path()
        return np.exp(-self.r * self.t) * max(
            0, prices[-1] - strike
        )  # we're only concerned with the final value of the underlying

    def european_put_payoff(self, strike):
        _, prices = self.simulate_price_path()
        return np.exp(-self.r * self.t) * max(
            0, strike - prices[-1]
        )  # terminal value used for calculating the payoff


def plot_paths(model, num_paths=10):
    """
    TODO:
    Use matplotlib to plot the movement of the underlying with a default of 10 simulations
    """

    for _ in range(num_paths):
        times, prices = model.simulate_price_path()
        times_1, log_returns = model.simulate_returns_path()
        pyplot.plot(times, prices)
        pyplot.plot(times_1, log_returns)


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
