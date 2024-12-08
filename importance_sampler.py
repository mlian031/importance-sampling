from mertonjdm import MertonJDM, SimulationResults
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ImportanceSamplingResults:
    standard_mc_price: float
    standard_mc_stderr: float
    is_price: float
    is_stderr: float
    variance_reduction: float


class ImportanceSampler:
    def __init__(self, model: MertonJDM) -> None:
        self.model = model

    def simulate_terminal_paths_standard(self, num_paths) -> np.ndarray:
        """Standard Monte Carlo simulation for terminal values"""
        return self.model.simulate_terminal_paths(num_paths)

    def simulate_terminal_paths_importance(self, num_paths, lambda_param):
        """
        Importance sampling simulation with modified drift
        Returns terminal log values and likelihood ratios
        """
        # Generate standard normal random variables
        Z = np.random.normal(0, 1, num_paths)

        # Brownian motion with modified drift
        W_T = Z * np.sqrt(self.model.T)

        # Generate Poisson random variables for jump counts
        N = np.random.poisson(self.model.lambda_j * self.model.T, num_paths)

        # Handle jumps
        sum_Y = np.zeros(num_paths)
        jump_paths = N > 0
        positive_jump_paths = N[jump_paths]
        sum_Y[jump_paths] = np.random.normal(
            loc=self.model.mu_j * positive_jump_paths,
            scale=np.sqrt(positive_jump_paths) * self.model.sigma_j,
        )

        # Modified risk-neutral drift for importance sampling
        risk_neutral_drift = (
            lambda_param
            - 0.5 * self.model.sigma**2
            - self.model.lambda_j
            * (np.exp(self.model.mu_j + 0.5 * self.model.sigma_j**2) - 1)
        )

        drift = risk_neutral_drift * self.model.T
        diffusion = self.model.sigma * W_T + sum_Y
        X_T = np.log(self.model.S0) + drift + diffusion

        # Calculate likelihood ratio
        likelihood_ratios = np.exp(
            -(lambda_param - self.model.r) * W_T / self.model.sigma
            - 0.5
            * ((lambda_param - self.model.r) / self.model.sigma) ** 2
            * self.model.T
        )

        return X_T, likelihood_ratios

    def optimize_lambda(self, num_paths: int):
        """
        Optimize the lambda parameter for importance sampling
        """

        def objective(lambda_param):
            result = self.compare_methods(num_paths, lambda_param[0])
            return -result.variance_reduction

        # Set bounds around risk-free rate
        lambda_lower = max(0.001, self.model.r - 2 * self.model.sigma)
        lambda_upper = self.model.r + 2 * self.model.sigma

        # Try multiple starting points
        best_result = float("inf")
        best_lambda = None
        starting_points = np.linspace(lambda_lower, lambda_upper, 10)

        for start_point in starting_points:
            result = minimize(
                objective,
                x0=[start_point],
                bounds=[(lambda_lower, lambda_upper)],
                method="L-BFGS-B",
            )

            if result.fun < best_result:
                best_result = result.fun
                best_lambda = result.x[0]

        final_result = self.compare_methods(num_paths, best_lambda)
        print("\n============================")
        print(f"Optimal Drift: {best_lambda}")
        print("\n============================")
        return best_lambda, final_result.variance_reduction

    def compare_methods(self, num_paths: int, lambda_param: float) -> ImportanceSamplingResults:
        """
        Compare standard Monte Carlo with importance sampling for option pricing.

        Parameters
        ----------
        num_paths : int
            Number of simulation paths to use.
        lambda_param : float
            The importance sampling drift parameter.

        Returns
        -------
        ImportanceSamplingResults
            A dataclass containing:
            - standard_mc_price : float
                Option price from standard Monte Carlo
            - standard_mc_stderr : float
                Standard error from standard Monte Carlo
            - is_price : float
                Option price from importance sampling
            - is_stderr : float
                Standard error from importance sampling
            - variance_reduction : float
                Variance reduction ratio achieved

        Notes
        -----
        The variance reduction ratio is calculated as (stderr_std/stderr_is)^2,
        where a higher value indicates better performance of importance sampling.
        """
        # Standard Monte Carlo
        terminal_values_std = self.simulate_terminal_paths_standard(num_paths)
        payoffs_std = self.model.compute_call_payoff(terminal_values_std, self.model.K)
        price_std = np.exp(-self.model.r * self.model.T) * np.mean(payoffs_std)
        stderr_std = (
            np.std(payoffs_std)
            * np.exp(-self.model.r * self.model.T)
            / np.sqrt(num_paths)
        )

        # Importance Sampling
        terminal_values_is, lr = self.simulate_terminal_paths_importance(
            num_paths, lambda_param
        )
        payoffs_is = self.model.compute_call_payoff(terminal_values_is, self.model.K)
        weighted_payoffs = payoffs_is * lr
        price_is = np.exp(-self.model.r * self.model.T) * np.mean(weighted_payoffs)
        stderr_is = (
            np.std(weighted_payoffs)
            * np.exp(-self.model.r * self.model.T)
            / np.sqrt(num_paths)
        )

        # Calculate variance reduction ratio
        var_reduction = (stderr_std / stderr_is) ** 2

        return ImportanceSamplingResults(
            standard_mc_price=price_std,
            standard_mc_stderr=stderr_std,
            is_price=price_is,
            is_stderr=stderr_is,
            variance_reduction=var_reduction,
        )
