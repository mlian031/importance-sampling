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
        1, 2, figsize=(12, 6)
    )  # merge the two plots

    for _ in range(num_paths):
        times, prices = model.simulate_price_path()
        price_plot.plot(times, prices)

    for _ in range(num_paths):
        times, log_returns = model.simulate_returns_path()
        # use cumulative sum of log returns for the plot
        cumulative_log_returns = np.cumsum(np.insert(log_returns, 0, 0))
        log_returns_plot.plot(times, cumulative_log_returns)

    price_plot.set_title("Stock Price vs Time")
    price_plot.set_xlabel("Time")
    price_plot.set_ylabel("Stock Price")

    log_returns_plot.set_title("Cumulative Log Returns vs Time")
    log_returns_plot.set_xlabel("Time")
    log_returns_plot.set_ylabel("Cumulative Log Returns")

    pyplot.tight_layout()
    fig.savefig("sample_paths.png")
    pyplot.close(fig)


def plot_option_price_convergence(model, strike, num_simulations=10000):
    """
    TODO:
    Use matplotlib to plot the convergence of
    the option price with 95% confidence intervals
    """
    call_payoffs = np.zeros(num_simulations)
    put_payoffs = np.zeros(num_simulations)
    for i in range(num_simulations):
        call_payoffs[i] = model.european_call_payoff(strike)
        put_payoffs[i] = model.european_put_payoff(strike)

    call_cumulative_mean = np.cumsum(call_payoffs) / (
        np.arange(num_simulations) + 1
    )
    put_cumulative_mean = np.cumsum(put_payoffs) / (
        np.arange(num_simulations) + 1
    )

    call_cumulative_stdv = np.sqrt(
        np.cumsum((call_payoffs - call_cumulative_mean) ** 2)
        / (np.arange(num_simulations) + 1)
    )
    put_cumulative_stdv = np.sqrt(
        np.cumsum((put_payoffs - put_cumulative_mean) ** 2)
        / (np.arange(num_simulations) + 1)
    )

    # calculate confidence intervals at logarithmically spaced points
    num_points = 20
    points = np.logspace(0, np.log10(num_simulations), num_points).astype(int)

    fig, (call_plot, put_plot) = pyplot.subplots(1, 2, figsize=(16, 8))

    call_plot.semilogx(
        range(1, num_simulations + 1), call_cumulative_mean, label="Mean"
    )  # change the x axis to log scale
    call_plot.errorbar(
        points,
        call_cumulative_mean[points - 1],
        yerr=1.96 * call_cumulative_stdv[points - 1] / np.sqrt(points),
        fmt="o",
        capsize=5,
        color="red",
        label="95% CI",
    )

    put_plot.semilogx(
        range(1, num_simulations + 1), put_cumulative_mean, label="Mean"
    )
    put_plot.errorbar(
        points,
        put_cumulative_mean[points - 1],
        yerr=1.96 * put_cumulative_stdv[points - 1] / np.sqrt(points),
        fmt="o",
        capsize=5,
        color="red",
        label="95% CI",
    )

    call_plot.set_title(
        "Call Option Price Convergence\nwith 95% Confidence Intervals",
        fontsize=10,
    )
    put_plot.set_title(
        "Put Option Price Convergence\nwith 95% Confidence Intervals",
        fontsize=10,
    )

    for ax in [call_plot, put_plot]:
        ax.set_ylabel("Estimated Option Price")
        ax.set_xlabel("Number of Simulations")
        ax.set_xlim(1, num_simulations)
        ax.legend()

    param_text = (
        f"S0: {model.s0:.2f}, strike: {strike:.2f}\n "
        f"r: {model.r:.2f}, μ: {model.mu:.2f}\n"
        f"λ: {model.lambda_:.2f}, σ: {model.sigma:.2f}\n"
        f"a: {model.a:.2f}, b: {model.b:.2f}\n"
        f"T: {model.t:.2f}, Steps: {model.steps}"
    )

    fig.text(
        0.82,
        0.85105,
        param_text,
        transform=fig.transFigure,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.text(
        0.3247,
        0.85105,
        param_text,
        transform=fig.transFigure,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    pyplot.tight_layout()
    fig.savefig("options_convergence_95_ci.png")
    pyplot.close(fig)


def main():
    print("Merton Jump Diffusion Model Simulation")

    model = MertonJumpDiffusionModel(
        s0=147.48,
        r=0.05,
        mu=0.05,
        lambda_=1.05,
        sigma=0.2,
        a=-0.08,
        b=0.4,
        t=1.0,
        steps=252,
    )

    strike = 150.0

    plot_paths(model)
    plot_option_price_convergence(model, strike)


if __name__ == "__main__":
    main()
