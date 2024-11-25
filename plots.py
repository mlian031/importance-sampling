import numpy as np
import matplotlib.pyplot as plt


def create_convergence_plot_with_ci(
    path_counts: np.ndarray,
    mean_estimates: np.ndarray,
    std_errors: np.ndarray,
    analytical_price: float,
    execution_times: np.ndarray,
) -> None:
    """
    Create plots showing price convergence with confidence intervals and performance against number of paths.

    Args:
        path_counts: Array of path counts used
        mean_estimates: Array of mean price estimates across M repetitions
        std_errors: Array of standard errors for the M repetitions
        analytical_price: Merton's analytical price
        execution_times: Array of execution times for each path count
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot 1: Price Convergence with Confidence Intervals
    relative_errors = np.abs(mean_estimates - analytical_price) / analytical_price * 100

    # Main convergence plot with confidence intervals
    ax1.semilogx(
        path_counts, mean_estimates, "bo-", label="Monte Carlo Estimate", markersize=4
    )
    ax1.fill_between(
        path_counts,
        mean_estimates - 1.96 * std_errors,
        mean_estimates + 1.96 * std_errors,
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    ax1.axhline(
        y=analytical_price, color="red", linestyle="--", label="Analytical Price"
    )

    # Add relative error on secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.semilogx(
        path_counts, relative_errors, "g--", alpha=0.5, label="Relative Error"
    )
    ax1_twin.set_ylabel("Relative Error (%)", color="g")

    # Set y-axis limits for price with some padding
    price_range = np.max(mean_estimates + 1.96 * std_errors) - np.min(
        mean_estimates - 1.96 * std_errors
    )
    ax1.set_ylim(
        [
            np.min(mean_estimates - 1.96 * std_errors) - 0.1 * price_range,
            np.max(mean_estimates + 1.96 * std_errors) + 0.1 * price_range,
        ]
    )

    # Customize first plot
    ax1.set_xlabel("Number of Paths (N)")
    ax1.set_ylabel("Option Price")
    ax1.set_title("Price Convergence with 95% Confidence Intervals")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Add grid
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Plot 2: Execution Time
    ax2.loglog(path_counts, execution_times, "ro-", label="Execution Time")

    # Calculate and plot theoretical O(N) complexity
    reference_time = execution_times[0] * (path_counts / path_counts[0])
    ax2.loglog(path_counts, reference_time, "k--", label="O(N) Reference", alpha=0.5)

    # Customize second plot
    ax2.set_xlabel("Number of Paths (N)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Computational Cost")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("merton_jdm_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
