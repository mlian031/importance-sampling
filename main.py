import numpy as np
from mertonjdm import MertonJDM
from importance_sampler import ImportanceSampler
from tabulate import tabulate
import time
from plots import create_convergence_plot_with_ci


def run_analysis():
    # Model parameters
    S0 = 100.0
    K = 150.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    mu = r

    lambda_j = 0.0
    mu_j = 0.0
    sigma_j = 0.0

    n_steps = 252
    budget = int(1e6)

    model = MertonJDM(S0, r, sigma, mu, T, K, n_steps, lambda_j, sigma_j, mu_j)
    importance_sampler = ImportanceSampler(model)

    # Optimize lambda parameter for importance sampling
    optimal_lambda, _ = importance_sampler.optimize_lambda(1000)

    path_counts = np.logspace(4, 5, 20).astype(int)
    results_data = []

    print("Running simulations...")

    analytical_price = model.closed_form_price()

    mean_estimates = np.zeros(len(path_counts))
    std_errors = np.zeros(len(path_counts))
    execution_times = np.zeros(len(path_counts))
    rmse_values = np.zeros(len(path_counts))

    is_mean_estimates = np.zeros(len(path_counts))
    is_std_errors = np.zeros(len(path_counts))
    is_execution_times = np.zeros(len(path_counts))
    is_rmse_values = np.zeros(len(path_counts))
    variance_reductions = np.zeros(len(path_counts))

    for i, N in enumerate(path_counts):
        M = int(budget / N)
        print(f"\nSimulating with N={N} paths, M={M} repetitions")

        # Standard Monte Carlo
        start_time = time.time()
        price_estimates = np.zeros(M)
        for m in range(M):
            terminal_log_returns = model.simulate_terminal_paths(N)
            payoffs = model.compute_call_payoff(terminal_log_returns, model.K)
            price_estimates[m] = np.exp(-model.r * model.T) * np.mean(payoffs)

        mean_price = np.mean(price_estimates)
        std_error = np.std(price_estimates, ddof=1) / np.sqrt(M)
        elapsed_time = time.time() - start_time
        rmse = np.sqrt(np.mean((price_estimates - analytical_price) ** 2))

        # Importance Sampling
        is_start_time = time.time()
        is_price_estimates = np.zeros(M)
        for m in range(M):
            terminal_values, lr = importance_sampler.simulate_terminal_paths_importance(
                N, optimal_lambda
            )
            payoffs = model.compute_call_payoff(terminal_values, model.K)
            weighted_payoffs = payoffs * lr
            is_price_estimates[m] = np.exp(-model.r * model.T) * np.mean(
                weighted_payoffs
            )

        is_mean_price = np.mean(is_price_estimates)
        is_std_error = np.std(is_price_estimates, ddof=1) / np.sqrt(M)
        is_elapsed_time = time.time() - is_start_time
        is_rmse = np.sqrt(np.mean((is_price_estimates - analytical_price) ** 2))
        variance_reduction = (std_error / is_std_error) ** 2

        # Store results
        mean_estimates[i] = mean_price
        std_errors[i] = std_error
        execution_times[i] = elapsed_time
        rmse_values[i] = rmse

        is_mean_estimates[i] = is_mean_price
        is_std_errors[i] = is_std_error
        is_execution_times[i] = is_elapsed_time
        is_rmse_values[i] = is_rmse
        variance_reductions[i] = variance_reduction

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
                f"{is_mean_price:.4f}",
                f"{is_std_error:.4f}",
                f"{is_rmse:.4f}",
                f"{is_elapsed_time:.2f}",
                f"{abs(is_mean_price - analytical_price) / analytical_price * 100:.2f}%",
                f"{variance_reduction:.2f}x",
            ]
        )

    create_convergence_plot_with_ci(
        path_counts,
        mean_estimates,
        std_errors,
        is_mean_estimates,
        is_std_errors,
        analytical_price,
        execution_times,
        is_execution_times,
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

    print("\nResults Summary:")
    print(tabulate(results_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run_analysis()
