import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from dataclasses import dataclass


@dataclass
class SimulationResults:
    mean_price: float
    std_error: float
    individual_prices: np.ndarray
    individual_std_errors: np.ndarray


class MertonJDM:
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
        Computes the closed form price using the black scholes model
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return float(
            self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        )

    def simulate_terminal_paths(self, num_paths):
        """
        Simulates terminal values for jump diffusion paths using vectorized calculations.
        Returns log of terminal values.

        Note: For risk-neutral pricing, we use r instead of mu in the drift term.
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

    def compute_call_payoff(self, X_T, K):
        """
        Computes discounted call option payoff from log terminal values.
        """
        S_T = np.exp(X_T)
        payoff = np.maximum(S_T - K, 0)
        return payoff

    def estimate_with_n_paths(self, N: int, M: int):
        price_estimates = np.zeros(M)

        for m in range(M):
            terminal_log_returns = self.simulate_terminal_paths(N)

            payoffs = self.compute_call_payoff(terminal_log_returns, self.K)

            price_estimates[m] = np.exp(-self.r * self.T) * np.mean(payoffs)

        return price_estimates

    def closed_form_price(self, n_terms=1000):
        """
        Computes the closed-form price using Merton's jump-diffusion formula.

        Reference:
        Robert C. Merton,
        Option pricing when underlying stock returns are discontinuous,
        Journal of Financial Economics,
        Volume 3, Issues 1–2,
        1976,
        Pages 125-144
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
