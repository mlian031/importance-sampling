import numpy as np
from mertonjdm import MertonJDM
from importance_sampler import ImportanceSampler
from tabulate import tabulate
import time
from plots import create_convergence_plot_with_ci


def get_float_input(prompt: str, default: float) -> float:
    """
    Get a float input from the user with a default value.

    Parameters
    ----------
    prompt : str
        The prompt message to display to the user.
    default : float
        The default value to use if no input is provided.

    Returns
    -------
    float
        The user input value or the default value.
    """
    value = input(f"{prompt} [{default}]: ").strip()
    return float(value) if value else default


def get_int_input(prompt: str, default: int) -> int:
    """
    Get an integer input from the user with a default value.

    Parameters
    ----------
    prompt : str
        The prompt message to display to the user.
    default : int
        The default value to use if no input is provided.

    Returns
    -------
    int
        The user input value or the default value.
    """
    value = input(f"{prompt} [{default}]: ").strip()
    return int(value) if value else default


def run_analysis():
    """
    Run the Monte Carlo simulation analysis for option pricing.
    
    This function performs the following steps:
    1. Collects model parameters from user input
    2. Initializes the Merton Jump Diffusion Model
    3. Runs both standard Monte Carlo and Importance Sampling simulations
    4. Compares results and generates performance metrics
    5. Creates visualization plots
    6. Outputs tabulated results
    
    The analysis includes:
    - Price convergence analysis
    - Standard error comparison
    - Execution time measurement
    - Variance reduction calculation
    
    Results are displayed in a formatted table and saved in plots.
    """
    print("\nEnter model parameters (press Enter to use default value):")

    S0 = get_float_input("Initial stock price (S0)", 100.0)
    K = get_float_input("Strike price (K)", 100.0)
    r = get_float_input("Risk-free rate (r)", 0.05)
    T = get_float_input("Time to maturity in years (T)", 1.0)
    sigma = get_float_input("Volatility (sigma)", 0.2)
    mu = r  # Setting mu = r for risk-neutral pricing

    print("\nJump parameters:")
    lambda_j = get_float_input("Jump intensity (lambda_j)", 0.0)
    mu_j = get_float_input("Jump mean (mu_j)", 0.0)
    sigma_j = get_float_input("Jump volatility (sigma_j)", 0.0)

    print("\nSimulation parameters:")
    n_steps = get_int_input("Number of time steps (n_steps)", 252)
    budget = get_int_input("Simulation budget", int(1e6))
    R = get_int_input("Number of repetitions (R)", 100)

    model = MertonJDM(S0, r, sigma, mu, T, K, n_steps, lambda_j, sigma_j, mu_j)
    importance_sampler = ImportanceSampler(model)

    optimal_lambda, _ = importance_sampler.optimize_lambda(1000)
    path_counts = np.logspace(3, 5, 50).astype(int)
    results_data = []

    print("\nRunning simulations...")
    analytical_price = model.closed_form_price()

    mean_estimates = np.zeros((R, len(path_counts)))
    std_errors = np.zeros((R, len(path_counts)))
    execution_times = np.zeros((R, len(path_counts)))
    rmse_values = np.zeros((R, len(path_counts)))

    is_mean_estimates = np.zeros((R, len(path_counts)))
    is_std_errors = np.zeros((R, len(path_counts)))
    is_execution_times = np.zeros((R, len(path_counts)))
    is_rmse_values = np.zeros((R, len(path_counts)))
    variance_reductions = np.zeros((R, len(path_counts)))

    for i, N in enumerate(path_counts):
        # M = int(budget / N)
        M = 100
        print(f"\nSimulating with N={N} paths, M={M} repetitions, R={R} outer repetitions")

        for r in range(R):
            # Standard Monte Carlo
            start_time = time.time()
            price_estimates = np.zeros(M)
            for m in range(M):
                terminal_log_returns = model.simulate_terminal_paths(N)
                payoffs = model.compute_call_payoff(terminal_log_returns, model.K)
                price_estimates[m] = np.exp(-model.r * model.T) * np.mean(payoffs)

            mean_estimates[r, i] = np.mean(price_estimates)
            std_errors[r, i] = np.std(price_estimates, ddof=1) / np.sqrt(M)
            execution_times[r, i] = time.time() - start_time
            rmse_values[r, i] = np.sqrt(np.mean((price_estimates - analytical_price) ** 2))

            # Importance Sampling
            is_start_time = time.time()
            is_price_estimates = np.zeros(M)
            for m in range(M):
                terminal_values, lr = importance_sampler.simulate_terminal_paths_importance(
                    N, optimal_lambda
                )
                payoffs = model.compute_call_payoff(terminal_values, model.K)
                weighted_payoffs = payoffs * lr
                is_price_estimates[m] = np.exp(-model.r * model.T) * np.mean(weighted_payoffs)

            is_mean_estimates[r, i] = np.mean(is_price_estimates)
            is_std_errors[r, i] = np.std(is_price_estimates, ddof=1) / np.sqrt(M)
            is_execution_times[r, i] = time.time() - is_start_time
            is_rmse_values[r, i] = np.sqrt(np.mean((is_price_estimates - analytical_price) ** 2))
            variance_reductions[r, i] = (std_errors[r, i] / is_std_errors[r, i]) ** 2

        # Store average results for display
        mc_mean = np.mean(mean_estimates[:, i])
        mc_stderr = np.mean(std_errors[:, i])
        mc_rmse = np.mean(rmse_values[:, i])
        mc_time = np.mean(execution_times[:, i])
        
        is_mean = np.mean(is_mean_estimates[:, i])
        is_stderr = np.mean(is_std_errors[:, i])
        is_rmse = np.mean(is_rmse_values[:, i])
        is_time = np.mean(is_execution_times[:, i])
        var_red = np.mean(variance_reductions[:, i])

        results_data.append(
            [
                N,
                M,
                f"{analytical_price:.4f}",
                f"{mc_mean:.4f}",
                f"{mc_stderr:.4f}",
                f"{mc_rmse:.4f}",
                f"{mc_time:.2f}",
                f"{abs(mc_mean - analytical_price) / analytical_price * 100:.2f}%",
                f"{is_mean:.4f}",
                f"{is_stderr:.4f}",
                f"{is_rmse:.4f}",
                f"{is_time:.2f}",
                f"{abs(is_mean - analytical_price) / analytical_price * 100:.2f}%",
                f"{var_red:.2f}x",
            ]
        )

    params = {
        "S0": S0,
        "K": K,
        "r": mu,
        "T": T,
        "sigma": sigma,
        "mu": mu,
        "lambda_j": lambda_j,
        "mu_j": mu_j,
        "sigma_j": sigma_j,
        "n_steps": n_steps,
        "budget": budget,
    }

    create_convergence_plot_with_ci(
        path_counts,
        np.mean(mean_estimates, axis=0),
        np.mean(std_errors, axis=0),
        np.mean(is_mean_estimates, axis=0),
        np.mean(is_std_errors, axis=0),
        analytical_price,
        np.mean(execution_times, axis=0),
        np.mean(is_execution_times, axis=0),
        params,
    )

    headers = [
        "Paths (N)",
        "Reps (M)",
        "Analytical",
        "MC Price",
        "MC StdErr",
        "MC RMSE",
        "MC Time(s)",
        "MC Rel.Err",
        "IS Price",
        "IS StdErr",
        "IS RMSE",
        "IS Time(s)",
        "IS Rel.Err",
        "Var.Red",
    ]

    avg_variance_reduction = np.mean(variance_reductions)

    print("\nResults Summary:")
    print(tabulate(results_data, headers=headers, tablefmt="grid"))
    print(f"\nAverage Variance Reduction: {avg_variance_reduction:0.3f}x")


if __name__ == "__main__":
    run_analysis()
