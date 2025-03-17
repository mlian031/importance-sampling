"""
plots.py - Visualization module for Monte Carlo option pricing with importance sampling

This module provides visualization tools for comparing standard Monte Carlo and importance sampling
methods in option pricing. It generates publication-quality plots to analyze:
1. Probability density distributions
2. Convergence behavior
3. Error distributions
4. Variance reduction effectiveness

The plots are designed for academic publication with LaTeX integration and consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from optimizer import call_option_price_IS, find_optimal_parameters

def setup_style():
    """Configure matplotlib for publication-quality plots"""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'text.usetex': True,  # Use LaTeX rendering for text
        
        # Figure settings
        'figure.figsize': (5.5, 3.4),  # Golden ratio for default aspect ratio
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        
        # Axes settings
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        
        # Tick settings
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # Legend settings
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

# Set a fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Apply the style
setup_style()

# Define consistent color palette for all plots
PALETTE = {
    'standard_mc': '#0072B2',  # Blue - Standard Monte Carlo
    'importance_sampling': '#D55E00',  # Vermillion - Importance Sampling
    'reference': '#000000',  # Black - Reference lines
    'positive_payoff_p': '#56B4E9',  # Light blue - Positive payoff under P
    'positive_payoff_q': '#E69F00',  # Orange - Positive payoff under Q
}

def black_scholes_call(S0, K, r, T, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Uses the analytical Black-Scholes formula:
    C = S0 * N(d1) - K * exp(-rT) * N(d2)
    where N(.) is the standard normal CDF
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    sigma : float
        Volatility of the stock
        
    Returns:
    --------
    float
        The Black-Scholes call option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def compare_estimators(S0, K, r, T, sigma, mu, n_samples=10000, n_experiments=100):
    """
    Compare standard Monte Carlo with importance sampling for option pricing.
    
    Performs multiple experiments to compare:
    1. Price accuracy (vs Black-Scholes)
    2. Variance reduction
    3. Mean Square Error (MSE)
    
    The comparison uses the same random seeds for both methods to ensure
    fair comparison with reduced Monte Carlo noise.
    
    Parameters:
    -----------
    S0, K, r, T, sigma : float
        Standard option parameters
    mu : float
        Drift parameter under measure P
    n_samples : int
        Number of Monte Carlo paths per experiment
    n_experiments : int
        Number of repeated experiments for statistical analysis
        
    Returns:
    --------
    dict
        Dictionary containing results from both methods including:
        - Prices from both methods
        - MSE and variance metrics
        - Variance reduction factor
        - Optimal importance sampling parameters
    """
    # Find optimal parameters for importance sampling measure Q
    optimal_tilde_mu, optimal_tilde_sigma = find_optimal_parameters(
        S0, K, r, T, sigma, mu
    )
    print(
        f"Optimal parameters: tilde_mu = {optimal_tilde_mu:.6f}, tilde_sigma = {optimal_tilde_sigma:.6f}"
    )

    # Calculate analytical Black-Scholes price for comparison
    bs_price = black_scholes_call(S0, K, r, T, sigma)

    # Initialize arrays to store simulation results
    standard_mc_prices = np.zeros(n_experiments)  # Standard Monte Carlo prices
    is_prices = np.zeros(n_experiments)  # Importance sampling prices

    for i in range(n_experiments):
        # Standard Monte Carlo simulation
        np.random.seed(SEED + i)  # Ensure reproducibility while varying experiments
        
        # Calculate parameters for risk-neutral measure
        m = np.log(S0) + (r - 0.5 * sigma**2) * T  # Drift term under risk-neutral measure
        v = sigma**2 * T  # Variance term
        
        # Generate log-normal samples for stock price at maturity
        X_samples = np.random.normal(m, np.sqrt(v), n_samples)
        S_T_samples = np.exp(X_samples)
        
        # Calculate option payoffs and discounted mean
        payoffs = np.maximum(S_T_samples - K, 0)  # Call option payoff max(S_T - K, 0)
        standard_mc_prices[i] = np.mean(payoffs) * np.exp(-r * T)  # Discounted expectation

        # Importance Sampling simulation (using same seed for fair comparison)
        np.random.seed(SEED + i)
        is_price, _ = call_option_price_IS(
            S0, K, r, T, sigma, mu, optimal_tilde_mu, optimal_tilde_sigma, n_samples
        )
        is_prices[i] = is_price

    # Calculate error metrics
    standard_mc_mse = np.mean((standard_mc_prices - bs_price) ** 2)  # Mean Square Error
    is_mse = np.mean((is_prices - bs_price) ** 2)

    # Calculate variance of estimators
    standard_mc_var = np.var(standard_mc_prices)  # Variance of standard MC estimates
    is_var = np.var(is_prices)  # Variance of importance sampling estimates

    # Calculate variance reduction factor (how many times more efficient IS is)
    var_reduction = standard_mc_var / is_var

    # Compile all results into a dictionary
    results = {
        "bs_price": bs_price,  # Analytical Black-Scholes price
        "standard_mc_prices": standard_mc_prices,  # Array of standard MC estimates
        "is_prices": is_prices,  # Array of importance sampling estimates
        "standard_mc_mse": standard_mc_mse,  # MSE for standard MC
        "is_mse": is_mse,  # MSE for importance sampling
        "standard_mc_var": standard_mc_var,  # Variance of standard MC
        "is_var": is_var,  # Variance of importance sampling
        "var_reduction": var_reduction,  # Variance reduction factor
        "optimal_tilde_mu": optimal_tilde_mu,  # Optimal drift for IS
        "optimal_tilde_sigma": optimal_tilde_sigma,  # Optimal volatility for IS
        "n_samples": n_samples,  # Number of paths per experiment
        "n_experiments": n_experiments,  # Number of repeated experiments
    }

    return results


def plot_density_comparison(
    S0, K, r, T, sigma, mu, optimal_tilde_mu, optimal_tilde_sigma
):
    """
    Plot probability densities of S_T under different measures.
    
    Compares three probability densities:
    1. Under P (risk-neutral measure)
    2. Under optimal Q (importance sampling measure)
    3. Theoretical optimal density (proportional to payoff * P-density)
    
    The plot illustrates how importance sampling shifts probability mass
    to regions that contribute most to the option price.
    
    Parameters:
    -----------
    S0, K, r, T, sigma : float
        Standard option parameters
    mu : float
        Drift under measure P
    optimal_tilde_mu, optimal_tilde_sigma : float
        Optimal parameters for importance sampling measure Q
    
    Returns:
    --------
    matplotlib.pyplot
        Plot object for further customization if needed
    """
    # Calculate parameters for the risk-neutral measure P
    m_p = np.log(S0) + (r - 0.5 * sigma**2) * T  # Drift term under P
    v_p = sigma**2 * T  # Variance term under P

    # Calculate parameters for the importance sampling measure Q
    m_q = np.log(S0) + (optimal_tilde_mu - 0.5 * optimal_tilde_sigma**2) * T  # Drift under Q
    v_q = optimal_tilde_sigma**2 * T  # Variance under Q

    # Calculate standard deviations for range determination
    std_p = np.sqrt(v_p)
    std_q = np.sqrt(v_q)

    # Calculate expected values of S_T under both measures (for visualization range)
    expected_S_T_P = np.exp(m_p + v_p / 2)
    expected_S_T_Q = np.exp(m_q + v_q / 2)

    # Determine plot range to capture both distributions
    std_deviations = 4  # Number of standard deviations to show
    min_s = max(
        0.1,  # Ensure positive minimum stock price
        min(S0 * np.exp(-std_deviations * std_p), S0 * np.exp(-std_deviations * std_q)),
    )
    max_s = max(
        S0 * np.exp(std_deviations * std_p),
        S0 * np.exp(std_deviations * std_q),
        K * 1.5,  # Include strike price with margin
    )

    # Create stock price range for density calculation
    s_range = np.linspace(min_s, max_s, 10000)
    log_s_range = np.log(s_range)

    # Calculate probability densities
    # Note: divide by s_range to convert from log-normal to normal density
    p_density = norm.pdf(log_s_range, m_p, np.sqrt(v_p)) / s_range  # Density under P
    q_density = norm.pdf(log_s_range, m_q, np.sqrt(v_q)) / s_range  # Density under Q
    
    # Calculate theoretical optimal importance sampling density
    optimal_density = np.zeros_like(s_range)
    positive_payoff_idx = s_range > K  # Region where option has positive payoff
    
    # Calculate optimal density in positive payoff region
    if np.any(positive_payoff_idx):
        # Optimal density proportional to payoff * original density
        unnormalized = (s_range[positive_payoff_idx] - K) * p_density[positive_payoff_idx]
        # Normalize to make it a proper probability density
        normalization_constant = np.trapz(unnormalized, s_range[positive_payoff_idx])
        if normalization_constant > 0:
            optimal_density[positive_payoff_idx] = unnormalized / normalization_constant

    # Create figure with publication-quality sizing
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    
    # Plot densities with consistent styling
    ax.plot(s_range, p_density, color=PALETTE['standard_mc'], linestyle='-', 
            label=r'Density under $P$ (Risk-neutral)')
    ax.plot(s_range, q_density, color=PALETTE['importance_sampling'], linestyle='-', 
            label=r'Density under optimal $Q$')
    ax.plot(s_range, optimal_density, color='grey', linestyle=':', linewidth=1.0, 
            label=r'Theoretical optimal density')
    
    # Add vertical lines for key prices
    ax.axvline(x=K, color='green', linestyle='--', linewidth=1.2, 
            label=fr'Strike $K = {K}$')
    ax.axvline(x=S0, color=PALETTE['reference'], linestyle=':', linewidth=1.2, 
            label=fr'Initial price $S_0 = {S0}$')

    # Add better formatting
    ax.set_title(r'Probability Density Comparison of $S_T$')
    ax.set_xlabel(r'Stock Price at Maturity ($S_T$)')
    ax.set_ylabel(r'Density')
    ax.legend(framealpha=0.9, loc='upper left')
    
    # Add model parameters in a text box
    param_text = (
        r"$S_0=%g$, $K=%g$, $r=%.2f$, $T=%.1f$, $\sigma=%.2f$" 
        % (S0, K, r, T, sigma)
    )
    ax.text(0.5, 0.02, param_text, transform=ax.transAxes,
            horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Calculate the probability that S_T > K under each measure
    p_prob = 1 - norm.cdf(np.log(K), m_p, np.sqrt(v_p))
    q_prob = 1 - norm.cdf(np.log(K), m_q, np.sqrt(v_q))

    print(f"Probability S_T > K under P: {p_prob:.6f}")
    print(f"Probability S_T > K under optimal Q: {q_prob:.6f}")

    # Save the figure
    fig.savefig("figures/density_comparison.pdf", bbox_inches='tight')
    fig.savefig("figures/density_comparison.png", dpi=300, bbox_inches='tight')

    return plt


def plot_density_with_shaded_payoff(
    S0, K, r, T, sigma, mu, optimal_tilde_mu, optimal_tilde_sigma
):
    """
    Plot probability densities with shaded positive payoff regions.
    
    Similar to plot_density_comparison but adds shaded regions where S_T > K
    to visualize where the option has positive payoff. This helps understand
    how importance sampling concentrates sampling in important regions.
    
    The shaded regions represent:
    1. Positive payoff region under P (risk-neutral)
    2. Positive payoff region under optimal Q (importance sampling)
    
    Parameters:
    -----------
    Same as plot_density_comparison
    
    Returns:
    --------
    matplotlib.pyplot
        Plot object for further customization if needed
    """
    # Calculate parameters for the risk-neutral measure P
    m_p = np.log(S0) + (r - 0.5 * sigma**2) * T  # Drift term under P
    v_p = sigma**2 * T  # Variance term under P

    # Calculate parameters for the importance sampling measure Q
    m_q = np.log(S0) + (optimal_tilde_mu - 0.5 * optimal_tilde_sigma**2) * T  # Drift under Q
    v_q = optimal_tilde_sigma**2 * T  # Variance under Q

    # Calculate standard deviations for range determination
    std_p = np.sqrt(v_p)
    std_q = np.sqrt(v_q)

    # Create a wider range to capture both distributions fully
    std_deviations = 4
    min_s = max(
        0.1,
        min(S0 * np.exp(-std_deviations * std_p), S0 * np.exp(-std_deviations * std_q)),
    )
    max_s = max(
        S0 * np.exp(std_deviations * std_p),
        S0 * np.exp(std_deviations * std_q),
        K * 1.5,  # Ensure we include the strike price with some margin
    )

    # Stock price range for plotting
    s_range = np.linspace(min_s, max_s, 10000)
    log_s_range = np.log(s_range)

    # Calculate densities
    p_density = (
        norm.pdf(log_s_range, m_p, np.sqrt(v_p)) / s_range
    )  # Convert to S_T density
    q_density = norm.pdf(log_s_range, m_q, np.sqrt(v_q)) / s_range
    
    # Calculate theoretical optimal density (zero below K, proportional to payoff * p_density above K)
    optimal_density = np.zeros_like(s_range)
    positive_payoff_idx = s_range > K
    
    # In theory, the optimal density would be proportional to the payoff function times original density
    # For regions where payoff > 0
    if np.any(positive_payoff_idx):
        # Calculate unnormalized density: (S_T - K) * p(S_T)
        unnormalized = (s_range[positive_payoff_idx] - K) * p_density[positive_payoff_idx]
        # Normalize to make it a proper density
        normalization_constant = np.trapz(unnormalized, s_range[positive_payoff_idx])
        if normalization_constant > 0:  # Avoid division by zero
            optimal_density[positive_payoff_idx] = unnormalized / normalization_constant

    # Create figure with publication sizing
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # Find indices where S_T > K (positive payoff region)
    positive_payoff_idx = s_range > K

    # Plot the densities with consistent styling
    ax.plot(s_range, p_density, color=PALETTE['standard_mc'], linestyle='-', 
            label=r'Density under $P$ (Risk-neutral)')
    ax.plot(s_range, q_density, color=PALETTE['importance_sampling'], linestyle='-', 
            label=r'Density under optimal $Q$')
    ax.plot(s_range, optimal_density, color='grey', linestyle=':', linewidth=1.0, 
            label=r'Theoretical optimal density')

    # Shade the positive payoff region for P (Risk-neutral) density
    ax.fill_between(
        s_range[positive_payoff_idx],
        0,
        p_density[positive_payoff_idx],
        color=PALETTE['positive_payoff_p'],
        alpha=0.3,
        label=r'Positive payoff region under $P$',
    )

    # Shade the positive payoff region for Q (Importance sampling) density
    ax.fill_between(
        s_range[positive_payoff_idx],
        0,
        q_density[positive_payoff_idx],
        color=PALETTE['positive_payoff_q'],
        alpha=0.3,
        label=r'Positive payoff region under $Q$',
    )

    # Add vertical lines for strike and initial price
    ax.axvline(x=K, color='green', linestyle='--', linewidth=1.2, 
            label=fr'Strike $K = {K}$')
    ax.axvline(x=S0, color=PALETTE['reference'], linestyle=':', linewidth=1.2, 
            label=fr'Initial price $S_0 = {S0}$')

    # Improve formatting
    ax.set_title(r'Probability Density Comparison with Shaded Positive Payoff Regions')
    ax.set_xlabel(r'Stock Price at Maturity ($S_T$)')
    ax.set_ylabel(r'Density')
    ax.legend(loc='upper left', fontsize=7)

    # Calculate the probability that S_T > K under each measure
    p_prob = 1 - norm.cdf(np.log(K), m_p, np.sqrt(v_p))
    q_prob = 1 - norm.cdf(np.log(K), m_q, np.sqrt(v_q))

    # Add probability text with nicely formatted box
    ax.text(
        0.71,
        0.89,
        fr'$P(S_T > K)$ under $P$: {p_prob:.4f}' + '\n' +
        fr'$P(S_T > K)$ under $Q$: {q_prob:.4f}',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='lightgray'),
        fontsize=8
    )

    # Add model parameters in a text box
    param_text = (
        r"$S_0=%g$, $K=%g$, $r=%.2f$, $T=%.1f$, $\sigma=%.2f$" 
        % (S0, K, r, T, sigma)
    )
    ax.text(0.5, 0.02, param_text, transform=ax.transAxes,
            horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Save the figure
    fig.savefig("figures/density_comparison_shaded.pdf", bbox_inches='tight')
    fig.savefig("figures/density_comparison_shaded.png", dpi=300, bbox_inches='tight')

    return plt


def plot_results(results):
    """
    Create a comprehensive visualization of the comparison results.
    
    Generates a two-panel figure showing:
    1. Histogram of price estimates from both methods
    2. Box plot of pricing errors relative to Black-Scholes
    
    The plots include:
    - Reference Black-Scholes price
    - Variance reduction metrics
    - Statistical summaries
    - Model parameters
    
    Parameters:
    -----------
    results : dict
        Dictionary containing comparison results from compare_estimators
    
    Returns:
    --------
    matplotlib.pyplot
        Plot object for further customization if needed
    """
    # Extract results from the dictionary
    bs_price = results["bs_price"]
    standard_mc_prices = results["standard_mc_prices"]
    is_prices = results["is_prices"]

    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6.5))

    # Plot 1: Histogram comparison of price estimates
    ax1 = axs[0]
    ax1.hist(standard_mc_prices, bins=20, alpha=0.6, color=PALETTE['standard_mc'], 
             label='Standard MC', edgecolor='white', linewidth=0.5)
    ax1.hist(is_prices, bins=20, alpha=0.6, color=PALETTE['importance_sampling'], 
             label='Importance Sampling', edgecolor='white', linewidth=0.5)
    ax1.axvline(x=bs_price, color='red', linestyle='--', linewidth=1.5, 
              label=fr'Black-Scholes ($C = {bs_price:.4f}$)')
    
    ax1.set_title(r'Distribution of Option Price Estimators', fontsize=10)
    ax1.set_xlabel(r'Option Price ($C$)')
    ax1.set_ylabel(r'Frequency')
    ax1.legend(fontsize=8)

    # Plot 2: Error distribution comparison
    ax2 = axs[1]
    mc_errors = standard_mc_prices - bs_price  # Errors for standard MC
    is_errors = is_prices - bs_price  # Errors for importance sampling

    # Create boxplot with custom styling
    boxprops_mc = dict(facecolor=PALETTE['standard_mc'], alpha=0.6)
    boxprops_is = dict(facecolor=PALETTE['importance_sampling'], alpha=0.6)
    
    # Generate boxplot with consistent styling
    bp = ax2.boxplot([mc_errors, is_errors], 
                     tick_labels=[r'Standard MC', r'Importance Sampling'],
                     patch_artist=True, widths=0.5, showfliers=True, 
                     flierprops=dict(marker='o', markersize=4, alpha=0.6))
    
    # Apply custom colors to boxplots
    bp['boxes'][0].set(facecolor=PALETTE['standard_mc'], alpha=0.6)
    bp['boxes'][1].set(facecolor=PALETTE['importance_sampling'], alpha=0.6)
    
    # Add reference line at zero error
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.2)
    ax2.set_title(r'Error Distribution', fontsize=10)
    ax2.set_ylabel(r'Error ($\hat{C} - C$)')

    # Add variance reduction information
    var_reduction = results['var_reduction']
    ax2.text(0.65, 0.05, fr'Variance reduction: {var_reduction:.2f}$\times$',
           transform=ax2.transAxes, fontsize=9,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Add model parameters as subtitle
    param_text = (
        r"Model Parameters: $S_0=%.1f$, $K=%.1f$, $r=%.2f$, $T=%.1f$, $\sigma=%.2f$, $M$(repetitions)$=%d$, $N$(paths)$=%d$" 
        % (results.get('S0', 100), results.get('K', 145), 
           results.get('r', 0.05), results.get('T', 1.0), 
           results.get('sigma', 0.2), results.get('n_experiments'), results.get('n_samples'))
    )
    fig.text(0.5, 0.01, param_text, ha='center', fontsize=8)
    
    # Adjust layout to prevent overlap
    fig.tight_layout(pad=1.5)

    # Print detailed statistics
    print("\nSummary Statistics:")
    print(f"Black-Scholes price: {bs_price:.6f}")
    print(
        f"Standard MC - Mean: {np.mean(standard_mc_prices):.6f}, Std: {np.std(standard_mc_prices):.6f}"
    )
    print(f"IS - Mean: {np.mean(is_prices):.6f}, Std: {np.std(is_prices):.6f}")
    print(f"Variance reduction factor: {results['var_reduction']:.2f}x")

    # Save plots in multiple formats
    fig.savefig("figures/estimator_comparison.pdf", bbox_inches='tight')
    fig.savefig("figures/estimator_comparison.png", dpi=300, bbox_inches='tight')

    return plt


def plot_convergence(S0, K, r, T, sigma, mu, n_experiments=100, log_scale=True):
    """
    Analyze and visualize the convergence behavior of both methods.
    
    Creates a two-panel figure showing:
    1. Price estimates vs number of samples (with error bars)
    2. Absolute error convergence analysis
    
    The second panel includes a reference line showing the theoretical
    Monte Carlo convergence rate of O(1/√N).
    
    Parameters:
    -----------
    S0, K, r, T, sigma : float
        Standard option parameters
    mu : float
        Drift under measure P
    n_experiments : int
        Number of repeated experiments for each sample size
    log_scale : bool
        Whether to use logarithmic scale for sample sizes
    
    Returns:
    --------
    matplotlib.pyplot
        Plot object for further customization if needed
    """
    # Calculate exact Black-Scholes price for reference
    bs_price = black_scholes_call(S0, K, r, T, sigma)
    
    # Find optimal importance sampling parameters
    optimal_tilde_mu, optimal_tilde_sigma = find_optimal_parameters(
        S0, K, r, T, sigma, mu
    )
    print(f"Optimal parameters: tilde_mu = {optimal_tilde_mu:.6f}, tilde_sigma = {optimal_tilde_sigma:.6f}")
    
    # Define range of sample sizes to test
    if log_scale:
        sample_sizes = np.logspace(3, 6, 15).astype(int)  # From 1,000 to 1,000,000
    else:
        sample_sizes = np.linspace(1000, 100000, 15).astype(int)
    
    # Initialize arrays for storing results
    mc_prices = np.zeros((len(sample_sizes), n_experiments))  # Standard MC results
    is_prices = np.zeros((len(sample_sizes), n_experiments))  # IS results
    mc_std_errs = np.zeros(len(sample_sizes))  # Standard errors for MC
    is_std_errs = np.zeros(len(sample_sizes))  # Standard errors for IS
    mc_abs_errors = np.zeros(len(sample_sizes))  # Absolute errors for MC
    is_abs_errors = np.zeros(len(sample_sizes))  # Absolute errors for IS

    # Run experiments for each sample size
    for i, n_samples in enumerate(sample_sizes):
        print(f"Running with {n_samples} samples...")
        
        for j in range(n_experiments):
            # Set seed for reproducibility while ensuring different experiments
            np.random.seed(SEED + i*100 + j)
            
            # Standard Monte Carlo simulation
            m = np.log(S0) + (r - 0.5 * sigma**2) * T  # Risk-neutral drift
            v = sigma**2 * T  # Variance
            X_samples = np.random.normal(m, np.sqrt(v), n_samples)
            S_T_samples = np.exp(X_samples)
            payoffs = np.maximum(S_T_samples - K, 0)
            mc_prices[i, j] = np.mean(payoffs) * np.exp(-r * T)
            
            # Importance Sampling simulation
            is_price, _ = call_option_price_IS(
                S0, K, r, T, sigma, mu, optimal_tilde_mu, optimal_tilde_sigma, n_samples
            )
            is_prices[i, j] = is_price
        
        # Calculate statistics across experiments
        mc_mean = np.mean(mc_prices[i])
        is_mean = np.mean(is_prices[i])
        
        # Calculate standard errors and absolute errors
        mc_std_errs[i] = np.std(mc_prices[i])  # Standard error for MC
        is_std_errs[i] = np.std(is_prices[i])  # Standard error for IS
        mc_abs_errors[i] = np.abs(mc_mean - bs_price)  # Absolute error for MC
        is_abs_errors[i] = np.abs(is_mean - bs_price)  # Absolute error for IS
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 7.0))
    
    # Plot 1: Price estimates with error bars
    ax1 = axs[0]
    mc_means = np.mean(mc_prices, axis=1)
    is_means = np.mean(is_prices, axis=1)
    
    # Plot estimates with error bars
    ax1.errorbar(sample_sizes, mc_means, yerr=mc_std_errs, fmt='o-', 
               capsize=4, color=PALETTE['standard_mc'], label='Standard Monte Carlo', 
               markersize=4, elinewidth=1.2)
    ax1.errorbar(sample_sizes, is_means, yerr=is_std_errs, fmt='o-', 
               capsize=4, color=PALETTE['importance_sampling'], label='Importance Sampling', 
               markersize=4, elinewidth=1.2)
    ax1.axhline(y=bs_price, color='red', linestyle='--', linewidth=1.2, 
               label=fr'Black-Scholes Price = {bs_price:.4f}')
    
    ax1.set_title(r'European Call Option: Monte Carlo Price Convergence')
    ax1.set_xlabel(r'Number of Simulated Paths ($N$)')
    ax1.set_ylabel(r'Option Price ($C$)')
    if log_scale:
        ax1.set_xscale('log')
    ax1.legend(loc='best', fontsize=8)
    
    # Plot 2: Absolute error convergence
    ax2 = axs[1]
    ax2.plot(sample_sizes, mc_abs_errors, 'o-', color=PALETTE['standard_mc'], 
            label='Standard Monte Carlo', markersize=4, linewidth=1.2)
    ax2.plot(sample_sizes, is_abs_errors, 'o-', color=PALETTE['importance_sampling'], 
            label='Importance Sampling', markersize=4, linewidth=1.2)
    
    # Add theoretical convergence rate reference line
    ref_line = sample_sizes**(-0.5) * mc_abs_errors[0] * np.sqrt(sample_sizes[0])
    ax2.plot(sample_sizes, ref_line, color='gray', linestyle='--', linewidth=1.2, 
            label=r'$1/\sqrt{N}$ Reference')
    
    ax2.set_title(r'Error Convergence Analysis')
    ax2.set_xlabel(r'Number of Simulated Paths ($N$)')
    ax2.set_ylabel(r'Absolute Error $|\hat{C} - C|$')
    if log_scale:
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=8)
    
    # Calculate final variance reduction factor
    largest_idx = -1  # Use largest sample size
    mc_var = mc_std_errs[largest_idx]**2
    is_var = is_std_errs[largest_idx]**2
    var_reduction = mc_var / is_var
    
    # Add model parameters as subtitle
    param_text = (
        r"Model Parameters: $S_0=%.1f$, $K=%.1f$, $r=%.2f$, $T=%.1f$, $\sigma=%.2f$, Repetitions=$%d$" 
        % (S0, K, r, T, sigma, n_experiments)
    )
    fig.text(0.5, 0.01, param_text, ha='center', fontsize=8)
    
    print(f"\nVariance reduction factor at N={sample_sizes[largest_idx]}: {var_reduction:.2f}x")
    
    # Adjust layout and save
    fig.tight_layout(pad=1.5)
    fig.savefig("figures/convergence_analysis.pdf", bbox_inches='tight')
    fig.savefig("figures/convergence_analysis.png", dpi=300, bbox_inches='tight')
    
    return plt

def run_plot():
    """
    Main function to generate all plots for the analysis.
    
    This function:
    1. Sets up the example parameters
    2. Runs the comparison analysis
    3. Generates all visualization plots
    4. Saves plots in both PDF and PNG formats
    
    The parameters are chosen to demonstrate the effectiveness of
    importance sampling for an out-of-the-money call option.
    """
    # Reset random seed for reproducibility
    np.random.seed(SEED)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    # Set example parameters for out-of-the-money call option
    S0 = 100.0  # Initial stock price
    K = 125.0  # Strike price (out-of-the-money)
    r = 0.05  # Risk-free rate
    T = 1.0  # Time to maturity
    sigma = 0.20  # Volatility under P
    mu = r  # Drift under P (risk-neutral setting)

    # Run comparison analysis
    results = compare_estimators(
        S0, K, r, T, sigma, mu, n_samples=100000, n_experiments=500
    )
    
    # Add model parameters to results for plotting
    results.update({'S0': S0, 'K': K, 'r': r, 'T': T, 'sigma': sigma})

    # Generate and save all plots
    plot_results(results).savefig(
        "figures/estimator_comparison.pdf", dpi=300, bbox_inches='tight'
    )
    
    convergence_plot = plot_convergence(S0, K, r, T, sigma, mu, n_experiments=500)
    convergence_plot.savefig("figures/convergence_analysis.pdf", dpi=300, bbox_inches='tight')

    density_plot = plot_density_comparison(
        S0, K, r, T, sigma, mu,
        results["optimal_tilde_mu"],
        results["optimal_tilde_sigma"],
    )
    density_plot.savefig("figures/density_comparison.pdf", dpi=300, bbox_inches='tight')

    shaded_density_plot = plot_density_with_shaded_payoff(
        S0, K, r, T, sigma, mu,
        results["optimal_tilde_mu"],
        results["optimal_tilde_sigma"],
    )
    shaded_density_plot.savefig("figures/density_comparison_shaded.pdf", dpi=300, bbox_inches='tight')

    print("Plots have been saved to the figures directory")


run_plot()
