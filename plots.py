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

    # Add parameter information as figure caption
    param_text = (
        "Parameters: S₀=100, K=150, r=0.05, T=1.0, σ=0.2, μ=0.05\n"
        "Jump Parameters: λ=1.0, μⱼ=-0.1, σⱼ=0.2\n"
        "Simulation: 252 steps/year, 1e6 budget"
    )
    fig.suptitle(param_text, y=1.02, fontsize=10)

    # Plot 1: Price Convergence with Error Bars
    relative_errors = np.abs(mean_estimates - analytical_price) / analytical_price * 100

    # Main convergence plot with error bars
    ax1.errorbar(
        path_counts,
        mean_estimates,
        yerr=1.96 * std_errors,  # 95% confidence intervals
        fmt="bo-",
        label="Monte Carlo Estimate",
        markersize=4,
        capsize=3,
        capthick=1,
        elinewidth=1,
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

    # Set x-axis to log scale
    ax1.set_xscale("log")

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
    ax2.loglog(
        path_counts, execution_times, "ro-", label="Execution Time", markersize=3
    )  # Smaller dots

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
    # Adjust figure margins to accommodate caption
    plt.subplots_adjust(top=0.95)
    plt.savefig("merton_jdm_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
