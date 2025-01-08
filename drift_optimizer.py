import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import differential_evolution

def european_call_payoff(S0, K, r, sigma, T, Z):
    """
    Given Z ~ N(0,1), return the European call payoff max(S_T - K,0).
    """
    ST = S0 * np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T)*Z)
    return np.maximum(ST - K, 0)

class DriftOptimizer:
    def __init__(self, S0, K, r, sigma, T, M):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.M = M
        self.bounds = [(-1e3, 1e3)]  # Bounds for the optimization
    
    def F(self, z):
        # Calculate the terminal stock price ST
        ST = self.S0 * np.exp(self.r * self.T - 0.5 * self.sigma**2 * self.T + self.sigma * np.sqrt(self.T) * z)
        if ST > self.K:
            return np.log(ST - self.K)
        else:
            return -1e6  # Large negative value for invalid domain
    
    def G(self, z):
        # Objective function to maximize
        return -(self.F(z) - 0.5 * z**2)  # Negative to maximize
    
    def optimize(self):
        # Optimize the objective function G using differential evolution
        result = differential_evolution(self.G, bounds=self.bounds)
        if result.success:
            self.z_opt = result.x[0]
            self.max_value = -result.fun  # Undo the negative sign
            return self.z_opt, self.max_value
        else:
            return None, None
    
    def plot_objective(self):
        # Plot the objective function G(z)
        z_vals = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)  # Adjust range if needed
        G_vals = [self.G(z) for z in z_vals]
        plt.plot(z_vals, G_vals)
        plt.xlabel("z")
        plt.ylabel("G(z)")
        plt.title("Objective Function G(z)")
        plt.show()

def black_scholes_call_price(S0, K, r, sigma, T):
    """
    Compute the analytical price of a European call option using the Black-Scholes formula.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying stock.
    T : float
        Time to maturity.
    
    Returns:
    --------
    float
        The Black-Scholes price of the call option.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Parameters for the European call
S0 = 100.0
K = 200.0
r = 0.05
sigma = 0.2
T = 1.0

M = 100  # Number of Monte Carlo simulations
N_values = np.logspace(4, 6, 90).astype(int)  # Different path counts for simulations

# Arrays to store price estimates and standard errors
mc_price_estimates = np.zeros(len(N_values))
is_price_estimates = np.zeros(len(N_values))
mc_stderrs = np.zeros(len(N_values))
is_stderrs = np.zeros(len(N_values))

optimal_mu = "Not found"

# Optimize the drift parameter
drift_optimizer = DriftOptimizer(S0, K, r, sigma, T, M)
optimal_mu, max_value = drift_optimizer.optimize()

# Monte Carlo and Importance Sampling simulations
for i, _ in enumerate(N_values):
    current_estimate_price_standard = []
    current_estimate_price_IS = []
    current_estimate_stderrs_standard = []
    current_estimate_stderrs_IS = []

    for _ in range(M):
        # Standard Monte Carlo simulation
        Z = np.random.randn(N_values[i])
        G_vals = european_call_payoff(S0, K, r, sigma, T, Z)
        discount_factor = np.exp(-r*T)
        price_standard = discount_factor * np.mean(G_vals)
        stderr_standard = discount_factor * np.std(G_vals, ddof=1)/np.sqrt(N_values[i])
        current_estimate_price_standard.append(price_standard)
        current_estimate_stderrs_standard.append(stderr_standard)

        # Importance Sampling simulation
        Z_prime = np.random.randn(N_values[i])
        Z_shifted = Z_prime + optimal_mu
        G_shifted = european_call_payoff(S0, K, r, sigma, T, Z_shifted)
        LR = np.exp(-optimal_mu*Z_prime - 0.5*optimal_mu**2)
        IS_estimates = G_shifted * LR
        price_IS = discount_factor * np.mean(IS_estimates)
        stderr_IS = discount_factor * np.std(IS_estimates, ddof=1)/np.sqrt(N_values[i])
        current_estimate_price_IS.append(price_IS)
        current_estimate_stderrs_IS.append(stderr_IS)

    # Store the mean estimates and standard errors
    mc_price_estimates[i] = np.mean(current_estimate_price_standard)
    mc_stderrs[i] = np.mean(current_estimate_stderrs_standard)
    is_price_estimates[i] = np.mean(current_estimate_price_IS)
    is_stderrs[i] = np.mean(current_estimate_stderrs_IS)

# Calculate the Black-Scholes price
bs_price = black_scholes_call_price(S0, K, r, sigma, T)

# Print data gathered
print("Path Count | MC Estimate | MC StdErr | IS Estimate | IS StdErr")
for i, N in enumerate(N_values):
    print(f"{N:9d} | {mc_price_estimates[i]:11.6f} | {mc_stderrs[i]:10.6f} | {is_price_estimates[i]:11.6f} | {is_stderrs[i]:10.6f}")

# Variance reduction calculation
variance_reduction_multiple = (np.mean(mc_stderrs) / np.mean(is_stderrs))**2
print(f"\nVariance Reduction Multiple: {variance_reduction_multiple:.2f}")

# Plot results
plt.figure(figsize=(10, 6))

# Top plot: Price estimates with CI
plt.errorbar(N_values, mc_price_estimates, yerr=1.96 * mc_stderrs, label='Monte Carlo', fmt='o', capsize=5)
plt.errorbar(N_values, is_price_estimates, yerr=1.96 * is_stderrs, label='Importance Sampling', fmt='o', capsize=5)
plt.axhline(bs_price, color='red', linestyle='--', label='Black-Scholes Analytical Price')
plt.xscale("log")
plt.xlabel('Number of Paths (N)')
plt.ylabel('Price Estimates')
plt.title('Monte Carlo vs. Importance Sampling with 95% CI')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("monte_carlo_vs_importance_sampling.png", dpi=600)
plt.show()
