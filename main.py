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
    R = get_int_input("Number of repetitions (R)", 500)
    model = MertonJDM(S0, r, sigma, mu, T, K, n_steps, lambda_j, sigma_j, mu_j)
    importance_sampler = ImportanceSampler(model)

    optimal_lambda, _ = importance_sampler.optimize_lambda(1000)
    path_counts = np.logspace(4, 5, 200).astype(int)
    results_data = []

    print("\nRunning simulations...")
    analytical_price = model.closed_form_price()

    # Initialize arrays for storing results
    arrays = {
        'standard': {
            'means': np.zeros((R, len(path_counts))),
            'stderr': np.zeros((R, len(path_counts))),
            'times': np.zeros((R, len(path_counts))),
            'rmse': np.zeros((R, len(path_counts)))
        },
        'importance': {
            'means': np.zeros((R, len(path_counts))),
            'stderr': np.zeros((R, len(path_counts))),
            'times': np.zeros((R, len(path_counts))),
            'rmse': np.zeros((R, len(path_counts)))
        },
        'variance_reduction': np.zeros((R, len(path_counts)))
    }

    for i, N in enumerate(path_counts):
        print(f"\nSimulating with N={N} paths, R={R} repetitions")

        for r in range(R):
            # Standard Monte Carlo
            start_time = time.time()
            terminal_log_returns = model.simulate_terminal_paths(N)
            payoffs = model.compute_call_payoff(terminal_log_returns, model.K)
            price = np.exp(-model.r * model.T) * np.mean(payoffs)
            stderr = np.exp(-model.r * model.T) * np.std(payoffs, ddof=1) / np.sqrt(N)

            arrays['standard']['means'][r, i] = price
            arrays['standard']['stderr'][r, i] = stderr
            arrays['standard']['times'][r, i] = time.time() - start_time
            arrays['standard']['rmse'][r, i] = np.abs(price - analytical_price)

            # Importance Sampling
            is_start_time = time.time()
            terminal_values, lr = importance_sampler.simulate_terminal_paths_importance(
                N, optimal_lambda
            )
            payoffs = model.compute_call_payoff(terminal_values, model.K)
            weighted_payoffs = payoffs * lr
            is_price = np.exp(-model.r * model.T) * np.mean(weighted_payoffs)
            is_stderr = np.exp(-model.r * model.T) * np.std(weighted_payoffs, ddof=1) / np.sqrt(N)

            arrays['importance']['means'][r, i] = is_price
            arrays['importance']['stderr'][r, i] = is_stderr
            arrays['importance']['times'][r, i] = time.time() - is_start_time
            arrays['importance']['rmse'][r, i] = np.abs(is_price - analytical_price)
            arrays['variance_reduction'][r, i] = (stderr / is_stderr) ** 2

        # Store results for display
        results_data.append([
            N,
            f"{analytical_price:.4f}",
            f"{np.mean(arrays['standard']['means'][:, i]):.4f}",
            f"{np.mean(arrays['standard']['stderr'][:, i]):.4f}",
            f"{np.mean(arrays['standard']['rmse'][:, i]):.4f}",
            f"{np.mean(arrays['standard']['times'][:, i]):.2f}",
            f"{abs(np.mean(arrays['standard']['means'][:, i]) - analytical_price) / analytical_price * 100:.2f}%",
            f"{np.mean(arrays['importance']['means'][:, i]):.4f}",
            f"{np.mean(arrays['importance']['stderr'][:, i]):.4f}",
            f"{np.mean(arrays['importance']['rmse'][:, i]):.4f}",
            f"{np.mean(arrays['importance']['times'][:, i]):.2f}",
            f"{abs(np.mean(arrays['importance']['means'][:, i]) - analytical_price) / analytical_price * 100:.2f}%",
            f"{np.mean(arrays['variance_reduction'][:, i]):.2f}x"
        ])

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
        np.mean(arrays['standard']['means'], axis=0),
        np.mean(arrays['standard']['stderr'], axis=0),
        np.mean(arrays['importance']['means'], axis=0),
        np.mean(arrays['importance']['stderr'], axis=0),
        analytical_price,
        np.mean(arrays['standard']['times'], axis=0),
        np.mean(arrays['importance']['times'], axis=0),
        params
    )

    headers = [
        "Paths (N)",
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
        "Var.Red"
    ]

    avg_variance_reduction = np.mean(arrays['variance_reduction'])

    print("\nResults Summary:")
    print(tabulate(results_data, headers=headers, tablefmt="grid"))
    print(f"\nAverage Variance Reduction: {avg_variance_reduction:0.3f}x")


if __name__ == "__main__":
    run_analysis()
