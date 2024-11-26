import numpy as np
from mertonjdm import MertonJDM
from tabulate import tabulate
import time
from plots import create_convergence_plot_with_ci


def run_analysis():
    # Model parameters
    S0 = 100.0  # Initial stock price
    K = 150.0  # Strike price
    r = 0.05  # Risk-free rate
    T = 1.0  # Time to maturity
    sigma = 0.2  # Volatility
    mu = r  # Risk-neutral drift

    # Jump parameters
    lambda_j = 0.0  # Jump intensity
    mu_j = 0.0  # Mean jump size
    sigma_j = 0.0  # Jump size volatility

    n_steps = 252  # Number of time steps
    budget = int(1e6)  # Budget for simulation

    model = MertonJDM(S0, r, sigma, mu, T, K, n_steps, lambda_j, sigma_j, mu_j)

    # Path configurations to test (logarithmically spaced)
    path_counts = np.logspace(4, 5, 20).astype(int)
    results_data = []

    print("Running simulations...")

    analytical_price = model.closed_form_price()

    mean_estimates = np.zeros(len(path_counts))
    std_errors = np.zeros(len(path_counts))
    execution_times = np.zeros(len(path_counts))
    rmse_values = np.zeros(len(path_counts))

    for i, N in enumerate(path_counts):
        M = int(budget / N)

        print(f"\nSimulating with N={N} paths, M={M} repetitions")
        start_time = time.time()

        price_estimates = np.zeros(M)

        # Repeat the valuation of the option M times
        for m in range(M):
            terminal_log_returns = model.simulate_terminal_paths(N)
            payoffs = model.compute_call_payoff(terminal_log_returns, model.K)
            price_estimates[m] = np.exp(-model.r * model.T) * np.mean(payoffs)

        mean_price = np.mean(price_estimates)
        std_error = np.std(price_estimates, ddof=1) / np.sqrt(M)

        squared_errors = (price_estimates - analytical_price) ** 2
        rmse = np.sqrt(np.mean(squared_errors))

        elapsed_time = time.time() - start_time

        mean_estimates[i] = mean_price
        std_errors[i] = std_error
        execution_times[i] = elapsed_time
        rmse_values[i] = rmse

        results_data.append(
            [
                N,
                M,
                f"{analytical_price:.4f}",
                f"{mean_price:.4f}",
                f"{std_error:.4f}",
                f"{rmse:.4f}",
                f"{elapsed_time:.2f}",
                f"{abs(mean_price - analytical_price) / analytical_price * 100:.2f}%",
            ]
        )

    create_convergence_plot_with_ci(
        path_counts, mean_estimates, std_errors, analytical_price, execution_times
    )

    # Table headers
    headers = [
        "Paths (N)",
        "Repetitions (M)",
        "Analytical Price",
        "Mean MC Price",
        "Std Error",
        "RMSE",
        "Time (s)",
        "Relative Error",
    ]

    print("\nResults Summary:")
    print(tabulate(results_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run_analysis()
