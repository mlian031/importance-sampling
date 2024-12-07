import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from dataclasses import dataclass


@dataclass
class SimulationResults:
    """
    Container for Monte Carlo simulation results.

    Attributes
    ----------
    mean_price : float
        Average option price across all simulations.
    std_error : float
        Standard error of the price estimate.
    individual_prices : ndarray
        Array of individual price estimates from each batch.
    individual_std_errors : ndarray
        Array of standard errors from each batch.
    """
    mean_price: float
    std_error: float
    individual_prices: np.ndarray
    individual_std_errors: np.ndarray


class MertonJDM:
    """
    Merton Jump Diffusion Model for option pricing.

    This class implements the Merton Jump Diffusion Model for pricing European options,
    incorporating both continuous diffusion and discrete jumps in asset prices.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the continuous component.
    mu : float
        Drift parameter.
    T : float
        Time to maturity in years.
    K : float
        Strike price.
    n_steps : int
        Number of time steps in the simulation.
    lambda_j : float
        Jump intensity (average number of jumps per year).
    sigma_j : float
        Standard deviation of jump size.
    mu_j : float
        Mean jump size.
    """

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        mu: float,
        T: float,
        K: float,
        n_steps: int,
        lambda_j: float,
        sigma_j: float,
        mu_j: float,
    ):
        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.T = float(T)
        self.K = float(K)
        self.n_steps = int(n_steps)
        self.lambda_j = float(lambda_j)
        self.sigma_j = float(sigma_j)
        self.mu_j = float(mu_j)
        self.dt = self.T / self.n_steps
        self.jump_mean = np.exp(float(mu_j) + 0.5 * float(sigma_j) ** 2)

    def black_scholes_price(self) -> float:
        """
        Compute the Black-Scholes price for a European call option.

        Returns
        -------
        float
            The Black-Scholes option price.

        Notes
        -----
        This method implements the standard Black-Scholes formula:
        C = S0 * N(d1) - K * exp(-rT) * N(d2)
        where N() is the standard normal cumulative distribution function.
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return float(
            self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )

    def simulate_terminal_paths(self, num_paths: int) -> np.ndarray:
        """
        Simulate terminal values for jump diffusion paths using vectorized calculations.

        Parameters
        ----------
        num_paths : int
            Number of paths to simulate.

        Returns
        -------
        ndarray
            Array of log terminal values for each path.

        Notes
        -----
        Uses risk-neutral pricing with drift adjusted for jumps.
        Combines both diffusion and jump components in a vectorized implementation.
        """

        # Generate normal random variables for diffusion
        Z = np.random.normal(0, 1, num_paths)

        # Brownian motion
        W_T = Z * np.sqrt(self.T)

        # Generate Poisson random variables for jump counts
        N = np.random.poisson(self.lambda_j * self.T, num_paths)

        sum_Y = np.zeros(num_paths)
        jump_paths = N > 0
        positive_jump_paths = N[jump_paths]
        sum_Y[jump_paths] = np.random.normal(
            loc=self.mu_j * positive_jump_paths,
            scale=np.sqrt(positive_jump_paths) * self.sigma_j,
        )

        risk_neutral_drift = (
            self.r
            - 0.5 * self.sigma**2
            - self.lambda_j * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1)
        )

        drift = risk_neutral_drift * self.T
        diffusion = self.sigma * W_T + sum_Y
        X_T = np.log(self.S0) + drift + diffusion

        return X_T

    def compute_call_payoff(self, X_T: np.ndarray, K: float) -> np.ndarray:
        """
        Compute discounted call option payoff from log terminal values.

        Parameters
        ----------
        X_T : ndarray
            Array of log terminal values.
        K : float
            Strike price.

        Returns
        -------
        ndarray
            Array of call option payoffs max(S_T - K, 0).
        """
        S_T = np.exp(X_T)
        payoff = np.maximum(S_T - K, 0)
        return payoff

    def estimate_with_n_paths(self, N: int, M: int) -> np.ndarray:
        """
        Estimate option prices using Monte Carlo simulation with multiple batches.

        Parameters
        ----------
        N : int
            Number of paths per batch.
        M : int
            Number of batches.

        Returns
        -------
        ndarray
            Array of M price estimates, one for each batch.
        """
        price_estimates = np.zeros(M)

        for m in range(M):
            terminal_log_returns = self.simulate_terminal_paths(N)

            payoffs = self.compute_call_payoff(terminal_log_returns, self.K)

            price_estimates[m] = np.exp(-self.r * self.T) * np.mean(payoffs)

        return price_estimates

    def closed_form_price(self, n_terms: int = 1000) -> float:
        """
        Compute the closed-form price using Merton's jump-diffusion formula.

        Parameters
        ----------
        n_terms : int, optional
            Number of terms to use in the series expansion, default 1000.

        Returns
        -------
        float
            The analytical option price.

        Notes
        -----
        Implementation based on:
        Robert C. Merton, "Option pricing when underlying stock returns are discontinuous",
        Journal of Financial Economics, Volume 3, Issues 1–2, 1976, Pages 125-144

        The formula extends the Black-Scholes model by incorporating a compound
        Poisson process for jumps.
        """
        lambda_prime = self.lambda_j * self.jump_mean
        price = 0

        for n in range(n_terms):
            sigma_n = np.sqrt(self.sigma**2 + n * self.sigma_j**2 / self.T)
            r_n = (
                self.r
                - self.lambda_j * (self.jump_mean - 1)
                + n * np.log(self.jump_mean) / self.T
            )

            d1 = (np.log(self.S0 / self.K) + (r_n + 0.5 * sigma_n**2) * self.T) / (
                sigma_n * np.sqrt(self.T)
            )
            d2 = d1 - sigma_n * np.sqrt(self.T)

            bs_price = self.S0 * norm.cdf(d1) - self.K * np.exp(
                -r_n * self.T
            ) * norm.cdf(d2)

            term = (
                np.exp(-lambda_prime * self.T)
                * (lambda_prime * self.T) ** n
                / factorial(n)
                * bs_price
            )
            price += term

        return price
