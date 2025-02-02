import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# -----------------------------
# Option and model parameters
# -----------------------------
S0    = 100.0   # initial stock price
K     = 145.0   # strike price
r     = 0.05    # risk-free rate
sigma = 0.2     # volatility
T     = 1.0     # time to maturity (in years)

sqrtT    = np.sqrt(T)
discount = np.exp(-r*T)
# Under risk-neutral dynamics the terminal stock price is:
#   S_T = S0 * exp((r - 0.5*sigma^2)T + sigma*sqrtT * Z)
# so let
A = S0 * np.exp((r - 0.5*sigma**2)*T)

# -----------------------------
# Black-Scholes Price
# -----------------------------
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call

bs_price = black_scholes_call(S0, K, r, sigma, T)
print("Black-Scholes Call Price =", bs_price)

# -----------------------------------------------------------
# Find the optimal drift shift (mu) using 2.8
# -----------------------------------------------------------
# For the call option the payoff is positive only when S_T > K.
# Writing S_T = A * exp(sigma*sqrtT*z) we must have:
#   A*exp(sigma*sqrtT*z) > K   =>   z > z_min, where:
z_min = np.log(K/A) / (sigma*sqrtT)

# Define the function f(z) = ln(A*exp(sigma*sqrtT*z) - K) - z^2/2.
# (Note: f(z) is only meaningful for z >= z_min. We return -inf below z_min.)
def f(z):
    if z < z_min:
        return -np.inf
    return np.log(A * np.exp(sigma*sqrtT*z) - K) - 0.5 * z**2

# To “maximize” f(z), we minimize -f(z) over [z_min, z_min+10] (an interval wide enough to capture the maximum)
res = minimize_scalar(lambda z: -f(z), bounds=(z_min, z_min+10), method='bounded')
mu_opt = res.x
print("Optimal drift shift (mu) for importance sampling =", mu_opt)

# ----------------------------------------------------------------------------------
# Set up simulation settings
# ----------------------------------------------------------------------------------
path_counts = np.logspace(3, 6, num=20, base=10, dtype=int)
num_rep    = 100

mc_estimates   = []
mc_std_errors  = []
is_estimates   = []
is_std_errors  = []
mc_variances   = []
is_variances   = []

# For each simulation size, repeat num_rep times and record the average price estimates.
for N in path_counts:
    mc_vals = np.zeros(num_rep)
    is_vals = np.zeros(num_rep)
    
    for i in range(num_rep):
        # ------------------------
        # Standard Monte Carlo
        # ------------------------
        # Sample Z ~ N(0,1) (shape: N)
        Z = np.random.randn(N)
        # Terminal stock price under risk–neutral measure:
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*sqrtT*Z)
        payoff = discount * np.maximum(S_T - K, 0)
        mc_vals[i] = np.mean(payoff)
        
        # ------------------------
        # Importance Sampling via Change of Drift
        # ------------------------
        # Instead of simulating Z, we simulate Z and use the shift mu_opt so that effectively we use Z+mu_opt.
        # (Recall: if Y ~ N(mu,1) then Y can be written as Z+mu with Z ~ N(0,1).)
        # The likelihood ratio is then: exp(-mu_opt*Z - 0.5*mu_opt**2)
        Z = np.random.randn(N)
        S_T_IS = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*sqrtT*(Z + mu_opt))
        likelihood = np.exp(-mu_opt*Z - 0.5*mu_opt**2)
        payoff_IS = discount * np.maximum(S_T_IS - K, 0) * likelihood
        is_vals[i] = np.mean(payoff_IS)
    
    # Compute average and standard error (over the 100 repetitions)
    mc_mean = np.mean(mc_vals)
    mc_std  = np.std(mc_vals, ddof=1)
    mc_se   = mc_std / np.sqrt(num_rep)
    
    is_mean = np.mean(is_vals)
    is_std  = np.std(is_vals, ddof=1)
    is_se   = is_std / np.sqrt(num_rep)
    
    mc_estimates.append(mc_mean)
    mc_std_errors.append(mc_se)
    is_estimates.append(is_mean)
    is_std_errors.append(is_se)
    
    mc_variances.append(mc_std**2)
    is_variances.append(is_std**2)
    
    print(f"\nNumber of paths: {N}")
    print(f"  Standard MC Price: {mc_mean:.4f} ± {1.96*mc_se:.4f} (95% CI), Variance: {mc_std**2:.8f}")
    print(f"  Importance Sampling Price: {is_mean:.4f} ± {1.96*is_se:.4f} (95% CI), Variance: {is_std**2:.8f}")
    if is_std**2 > 0:
        vr_ratio = mc_std**2 / is_std**2
        print(f"  Variance Reduction Ratio (MC/IS): {vr_ratio:.2f}")
    else:
        print("  Variance Reduction Ratio: Infinity (IS variance is zero)")

# -------------------------------------------------------
# Plotting: Option price estimates vs. number of paths
# -------------------------------------------------------
# We use a log-scale on the x-axis. The error bars will be 95% confidence intervals.

plt.style.use('classic')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'], 
    'text.usetex': True,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'figure.figsize': (8, 8),
    'figure.dpi': 300,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}'
})

# Convert results to numpy arrays for plotting
path_counts_arr = np.array(path_counts)
mc_est_arr      = np.array(mc_estimates)
mc_err_arr      = 1.96 * np.array(mc_std_errors)  # 95% CI
is_est_arr      = np.array(is_estimates)
is_err_arr      = 1.96 * np.array(is_std_errors)

fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

ax_params = fig.add_subplot(gs[0])
ax_params.axis('off')

params_text = (
    "Model Parameters:\n"
    f"Initial Stock Price ($S_0$): {S0:.2f}\n"
    f"Strike Price (K): {K:.2f}\n"
    f"Time to Maturity (T): {T:.1f} years\n"
    f"Risk-free Rate (r): {r*100:.1f}%\n"
    f"Volatility ($\\sigma$): {sigma*100:.1f}%\n"
    f"Repetitions: {num_rep}"
)

ax_params.text(0.5, 0.5, params_text, ha='center', va='center',
               transform=ax_params.transAxes,
               bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round,pad=0.5'))

ax = fig.add_subplot(gs[1])
colors = {
    'is': '#000080',   # Navy blue for importance sampling
    'std': '#8B0000',  # Dark red for standard MC
    'bs': '#006400'    # Dark green for Black-Scholes reference
}

# Plot Importance Sampling results
ax.errorbar(path_counts_arr, is_est_arr, yerr=is_err_arr,
            fmt='o-', capsize=3, capthick=1,
            label='Importance Sampling',
            color=colors['is'],
            markerfacecolor='white',
            markeredgewidth=1.2,
            ecolor=colors['is'],
            alpha=0.8)

# Plot Standard Monte Carlo results
ax.errorbar(path_counts_arr, mc_est_arr, yerr=mc_err_arr,
            fmt='s--', capsize=3, capthick=1,
            label='Standard Monte Carlo',
            color=colors['std'],
            markerfacecolor='white',
            markeredgewidth=1.2,
            ecolor=colors['std'],
            alpha=0.8)

ax.axhline(y=bs_price, color=colors['bs'], linestyle='-.',
           label=f'Black-Scholes Price = \\${bs_price:.2f}')

ax.set_xscale('log')
ax.set_xlim(path_counts_arr[0] * 0.8, path_counts_arr[-1] * 1.2)
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.grid(True, which='major', linestyle='-', alpha=0.25)
ax.grid(True, which='minor', linestyle=':', alpha=0.15)

ax.set_xlabel(r'Number of Simulated Paths ($N$)')
ax.set_ylabel(r'Option Price ($\mathcal{C}$)')
ax.set_title(r'European Call Option: Monte Carlo Price Convergence Analysis', pad=10)

leg = ax.legend(frameon=True, fancybox=False, edgecolor='black',
                bbox_to_anchor=(0.98, 0.98), loc='upper right', borderaxespad=0)
leg.get_frame().set_linewidth(0.8)

plt.savefig('option_pricing_mc.pdf', bbox_inches='tight', dpi=300)
plt.savefig('option_pricing_mc.png', bbox_inches='tight', dpi=300)
