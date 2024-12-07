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
    params: dict,
) -> None:
    """
    Create a three-panel plot showing convergence analysis of Monte Carlo simulations.

    Parameters
    ----------
    path_counts : ndarray
        Array of different path counts used in simulations.
    mean_estimates : ndarray
        Mean price estimates from standard Monte Carlo.
    std_errors : ndarray
        Standard errors from standard Monte Carlo.
    is_mean_estimates : ndarray
        Mean price estimates from Importance Sampling.
    is_std_errors : ndarray
        Standard errors from Importance Sampling.
    analytical_price : float
        The true analytical price for comparison.
    execution_times : ndarray
        Execution times for standard Monte Carlo simulations.
    is_execution_times : ndarray
        Execution times for Importance Sampling simulations.
    params : dict
        Dictionary containing model parameters for plot annotation.
        Must include: 'S0', 'K', 'r', 'T', 'sigma', 'mu', 'lambda_j',
        'mu_j', 'sigma_j', 'n_steps', 'budget'.

    Returns
    -------
    None
        Saves the plot to 'merton_jdm_analysis.png'.

    Notes
    -----
    Creates three subplots:
    1. Price convergence with confidence intervals
    2. Relative error convergence
    3. Computational cost comparison
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    param_text = (
        f"Parameters: S₀={params['S0']}, K={params['K']}, r={params['r']}, "
        f"T={params['T']}, σ={params['sigma']}, μ={params['mu']}\n"
        f"Jump Parameters: λ={params['lambda_j']}, μⱼ={params['mu_j']}, σⱼ={params['sigma_j']}\n"
        f"Simulation: {params['n_steps']} steps/year, {params['budget']} budget"
    )
    fig.suptitle(param_text, y=1.02, fontsize=10)

    # Price convergence plot
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
    ax1.legend(loc="best")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Relative error plot
    relative_errors = np.abs(mean_estimates - analytical_price) / analytical_price * 100
    is_relative_errors = (
        np.abs(is_mean_estimates - analytical_price) / analytical_price * 100
    )

    ax2.semilogx(path_counts, relative_errors, "bo-", label="Standard MC", markersize=4)
    ax2.semilogx(
        path_counts,
        is_relative_errors,
        "ro-",
        label="Importance Sampling",
        markersize=4,
    )
    ax2.set_xlabel("Number of Paths (N)")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title("Relative Error Convergence")
    ax2.legend(loc="best")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    # Computational cost plot with actual times
    ax3.loglog(path_counts, execution_times, "bo-", label="Standard MC", markersize=3)
    ax3.loglog(
        path_counts,
        is_execution_times,
        "ro-",
        label="Importance Sampling",
        markersize=3,
    )

    ax3.set_xlabel("Number of Paths (N)")
    ax3.set_ylabel("Execution Time (seconds)")
    ax3.set_title("Computational Cost")
    ax3.grid(True, which="both", linestyle="--", alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("merton_jdm_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()