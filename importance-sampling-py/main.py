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
        z = np.random.normal(0, 1, size=self.steps)
        # generate poisson random variable
        n = np.random.poisson(lam=(self.lambda_ * self.dt), size=self.steps)
        # generate jump component
        y = np.random.normal(loc=self.a, scale=self.b, size=self.steps)
        # get an array of n='steps' evenly spaced times
        t = np.linspace(
            0, self.t, self.steps + 1
        )  # splits it into n='steps' number of intervals

        log_returns = (
            (self.mu - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
            + n * y
        )

        return t, log_returns

    def simulate_price_path(self):
        t, log_returns = self.simulate_returns_path()

        log_prices = np.cumsum(log_returns)
        prices = self.s0 * np.exp(np.insert(log_prices, 0, 0))

        return t, prices

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
    Use matplotlib to plot the movement of the underlying
    with a default of 10 simulations
    """

    fig, (price_plot, log_returns_plot) = pyplot.subplots(
        1, 2, figsize=(12, 12)
    )

    for _ in range(num_paths):
        times, prices = model.simulate_price_path()
        price_plot.plot(times, prices)

    for _ in range(num_paths):
        times, log_returns = model.simulate_returns_path()
        log_returns_plot.plot(times, log_returns)

    price_plot.title("Stock Price vs Time")
    price_plot.xlabel("Time")
    price_plot.ylabel("Stock Price")

    log_returns_plot.title("Log Returns vs Time")
    log_returns_plot.xlabel("Time")
    log_returns_plot.ylabel("Log Returns")

    fig.savefig("sample_paths.png")


def plot_option_price_convergence(model, strike, num_simulations=10000):
    """
    TODO:
    Use matplotlib to plot the convergence of
    the option price with 95% confidence intervals
    """
    pass


def main():
    print("Merton Jump Diffusion Model Simulation")

    model = MertonJumpDiffusionModel(
        s0=100,
        r=0.05,
        mu=0.05,
        lambda_=1.0,
        sigma=0.2,
        a=-0.05,
        b=0.1,
        t=1.0,
        steps=252,
    )

    plot_paths(model)


if __name__ == "__main__":
    main()
