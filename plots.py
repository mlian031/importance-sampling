import numpy as np
import matplotlib.pyplot as plt


def create_convergence_plot_with_ci(
    path_counts: np.ndarray,
    mean_estimates: np.ndarray,
    std_errors: np.ndarray,
    is_mean_estimates: np.ndarray,
    is_std_errors: np.ndarray,
    analytical_price: float,
    execution_times: np.ndarray,
    is_execution_times: np.ndarray,
) -> None:
    """
    Create plots showing price convergence with confidence intervals and performance against number of paths.
    Includes both standard Monte Carlo and importance sampling results.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    param_text = (
        "Parameters: S₀=100, K=150, r=0.05, T=1.0, σ=0.2, μ=0.05\n"
        "Jump Parameters: λ=1.0, μⱼ=-0.1, σⱼ=0.2\n"
        "Simulation: 252 steps/year, 1e6 budget"
    )
    fig.suptitle(param_text, y=1.02, fontsize=10)

    # Calculate relative errors for both methods
    relative_errors = np.abs(mean_estimates - analytical_price) / analytical_price * 100
    is_relative_errors = (
        np.abs(is_mean_estimates - analytical_price) / analytical_price * 100
    )

    # Standard MC plot
    ax1.errorbar(
        path_counts,
        mean_estimates,
        yerr=1.96 * std_errors,
        fmt="bo-",
        label="Standard Monte Carlo",
        markersize=4,
        capsize=3,
        capthick=1,
        elinewidth=1,
    )

    # Importance Sampling plot
    ax1.errorbar(
        path_counts,
        is_mean_estimates,
        yerr=1.96 * is_std_errors,
        fmt="ro-",
        label="Importance Sampling",
        markersize=4,
        capsize=3,
        capthick=1,
        elinewidth=1,
    )

    ax1.axhline(
        y=analytical_price, color="green", linestyle="--", label="Analytical Price"
    )

    # Add relative error on secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.semilogx(
        path_counts,
        relative_errors,
        "b--",
        alpha=0.5,
        label="Standard MC Relative Error",
    )
    ax1_twin.semilogx(
        path_counts, is_relative_errors, "r--", alpha=0.5, label="IS Relative Error"
    )
    ax1_twin.set_ylabel("Relative Error (%)", color="black")

    # Set y-axis limits for price
    all_estimates = np.concatenate([mean_estimates, is_mean_estimates])
    all_errors = np.concatenate([std_errors, is_std_errors])
    price_range = np.max(all_estimates + 1.96 * all_errors) - np.min(
        all_estimates - 1.96 * all_errors
    )
    ax1.set_ylim(
        [
            np.min(all_estimates - 1.96 * all_errors) - 0.1 * price_range,
            np.max(all_estimates + 1.96 * all_errors) + 0.1 * price_range,
        ]
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Paths (N)")
    ax1.set_ylabel("Option Price")
    ax1.set_title("Price Convergence with 95% Confidence Intervals")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Execution Time Plot
    ax2.loglog(
        path_counts, execution_times, "bo-", label="Standard MC Time", markersize=3
    )
    ax2.loglog(path_counts, is_execution_times, "ro-", label="IS Time", markersize=3)

    reference_time = execution_times[0] * (path_counts / path_counts[0])
    ax2.loglog(path_counts, reference_time, "k--", label="O(N) Reference", alpha=0.5)

    ax2.set_xlabel("Number of Paths (N)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Computational Cost")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("merton_jdm_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
