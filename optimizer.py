import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def call_option_price_IS(
        S0: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        mu: float,
        tilde_mu: float,
        tilde_sigma: float,
        n_samples: 10000
):
    """
    Price a European call option using Importance Sampling with both drift and volatility optimization.

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
        Volatility under the real-world measure P
    mu : float
        Drift under the real-world measure P
    tilde_mu : float
        Drift under the importance sampling measure Q
    tilde_sigma : float
        Volatility under the importance sampling measure Q
    n_samples : int
        Number of Monte Carlo samples

    Returns:
    --------
    float
        Estimated option price
    float
        Standard error of the estimate
    """
    
    m =  np.log(S0) + (mu - 0.5 * sigma**2) * T # Drift under the real-world measure P
    v = sigma ** 2 * T # Variance under the real-world measure P

    # Define the parameters for log-stock price X under P
    tilde_m = (
        np.log(S0) + (tilde_mu - 0.5 * tilde_sigma**2) * T
    )
    tilde_v = tilde_sigma**2 * T

    # Generate samples under Q
    X_samples = np.random.normal(tilde_m, np.sqrt(tilde_v), n_samples)
    S_T_samples = np.exp(X_samples)

    # Calculate the payoffs under Q
    payoffs = np.maximum(S_T_samples - K, 0)

    # Likelihood ratio
    likelihood_ratios = np.sqrt(tilde_v / v) * np.exp(
        -((X_samples - m) ** 2) / (2 * v) + ((X_samples - tilde_m) ** 2) / (2 * tilde_v)
    )

    discount_factor = np.exp(-r * T)
    estimators = discount_factor * payoffs * likelihood_ratios

    option_price = np.mean(estimators) # Estimated option price
    standard_error = np.std(estimators) / np.sqrt(n_samples) # Standard error of the estimate

    return option_price, standard_error

def objective_function(params, S0, K, r, T, sigma, mu, n_samples=1000):
    """
    Objective function to be minimized: F(tilde_mu, tilde_sigma)

    This represents the second moment of the estimator as derived in Step 6 of the document.
    F(tilde_mu, tilde_sigma) = E_Q[g(e^X)^2 * 1_{e^X in E} * L(X)^2]
    """
    tilde_mu, tilde_sigma = params

    # Ensure tilde_sigma is positive
    if tilde_sigma <= 0:
        return 1e10

    m = np.log(S0) + (mu - 0.5 * sigma**2) * T
    v = sigma**2 * T

    tilde_m = np.log(S0) + (tilde_mu - 0.5 * tilde_sigma**2) * T
    tilde_v = tilde_sigma**2 * T

    X_samples = np.random.normal(tilde_m, np.sqrt(tilde_v), n_samples)
    S_T_samples = np.exp(X_samples)

    payoffs_squared = np.maximum(S_T_samples - K, 0) ** 2

    likelihood_ratios_squared = (tilde_v / v) * np.exp(
        -((X_samples - m) ** 2) / v + ((X_samples - tilde_m) ** 2) / tilde_v
    )

    discount_factor_squared = np.exp(-2 * r * T)

    # Calculate second moment (as per Step 6.1 and 6.3)
    second_moment = np.mean(
        discount_factor_squared * payoffs_squared * likelihood_ratios_squared
    )

    return second_moment


def find_optimal_parameters(S0, K, r, T, sigma, mu, initial_guess=None):
    """
    Find optimal tilde_mu and tilde_sigma to minimize the variance
    of the importance sampling estimator, as per Step 7.
    """
    if initial_guess is None:
        # Large deviations insight (Step 8): For out-of-the-money calls,
        # a reasonable initial guess is to adjust the drift to make the event typical
        log_moneyness = np.log(K / S0)

        initial_tilde_mu = (log_moneyness + 0.5 * sigma**2 * T) / T + 0.5 * sigma**2
        initial_tilde_sigma = sigma  # Start with original volatility

        initial_guess = [initial_tilde_mu, initial_tilde_sigma]

    # Minimize the objective function using a numerical optimizer (Step 7.2)
    result = minimize(
        objective_function,
        initial_guess,
        args=(S0, K, r, T, sigma, mu),
        method="Nelder-Mead",
        bounds=[(None, None), (1e-6, None)],  # Ensure positive volatility
    )

    return result.x


def black_scholes_call(S0, K, r, T, sigma):
    """Calculate Black-Scholes price for a European call option"""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def main():
    S0 = 100.0  # Initial stock price
    K = 120.0  # Strike price (out-of-the-money)
    r = 0.05  # Risk-free rate
    T = 1.0  # Time to maturity
    sigma = 0.2  # Volatility under P
    mu = r  # Drift under P (using risk-neutral measure)
    n_samples = 10000  # Number of samples per experiment
    n_experiments = 100  # Number of experiments for statistical stability

    print("Finding optimal importance sampling parameters...")
    optimal_params = find_optimal_parameters(S0, K, r, T, sigma, mu)
    optimal_tilde_mu, optimal_tilde_sigma = optimal_params
    print(f"Optimal tilde_mu: {optimal_tilde_mu:.6f}")
    print(f"Optimal tilde_sigma: {optimal_tilde_sigma:.6f}")

    bs_price = black_scholes_call(S0, K, r, T, sigma)
    print(f"Black-Scholes price: {bs_price:.6f}")

    standard_mc_prices = np.zeros(n_experiments)
    is_prices = np.zeros(n_experiments)
    standard_mc_errors = np.zeros(n_experiments)
    is_errors = np.zeros(n_experiments)
    var_reduction_factors = np.zeros(n_experiments)

    print(f"\nRunning {n_experiments} experiments with {n_samples} samples each...")
    
    for i in range(n_experiments):
        # Standard Monte Carlo
        m = np.log(S0) + (r - 0.5 * sigma**2) * T  # Risk-neutral measure
        v = sigma**2 * T
        X_samples = np.random.normal(m, np.sqrt(v), n_samples)
        S_T_samples = np.exp(X_samples)
        payoffs = np.maximum(S_T_samples - K, 0)
        discount_factor = np.exp(-r * T)
        
        standard_mc_prices[i] = discount_factor * np.mean(payoffs)
        standard_mc_errors[i] = discount_factor * np.std(payoffs) / np.sqrt(n_samples)

        # Importance Sampling
        is_price, is_error = call_option_price_IS(
            S0, K, r, T, sigma, mu, optimal_tilde_mu, optimal_tilde_sigma, n_samples
        )
        
        is_prices[i] = is_price
        is_errors[i] = is_error
        
        # Calculate variance reduction for this experiment
        var_reduction_factors[i] = (standard_mc_errors[i] / is_errors[i]) ** 2
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_experiments} experiments")

    standard_mc_mse = np.mean((standard_mc_prices - bs_price) ** 2)
    is_mse = np.mean((is_prices - bs_price) ** 2)

    standard_mc_var = np.var(standard_mc_prices)
    is_var = np.var(is_prices)

    overall_var_reduction = standard_mc_var / is_var

    print("\nResults after multiple experiments:")
    print(f"Black-Scholes price: {bs_price:.6f}")
    print(f"Standard MC - Mean: {np.mean(standard_mc_prices):.6f}, Std: {np.std(standard_mc_prices):.6f}")
    print(f"Importance Sampling - Mean: {np.mean(is_prices):.6f}, Std: {np.std(is_prices):.6f}")
    print(f"Standard MC - Average SE: {np.mean(standard_mc_errors):.6f}")
    print(f"Importance Sampling - Average SE: {np.mean(is_errors):.6f}")
    print(f"MSE - Standard MC: {standard_mc_mse:.8f}")
    print(f"MSE - Importance Sampling: {is_mse:.8f}")
    print(f"MSE reduction factor: {standard_mc_mse / is_mse:.2f}x")
    print(f"Overall variance reduction factor: {overall_var_reduction:.2f}x")
    print(f"Average variance reduction factor: {np.mean(var_reduction_factors):.2f}x")
    print(f"Min variance reduction factor: {np.min(var_reduction_factors):.2f}x")
    print(f"Max variance reduction factor: {np.max(var_reduction_factors):.2f}x")


def norm_cdf(x):
    """Standard normal cumulative distribution function"""
    return norm.cdf(x)
