# Importance Sampling

The goal of this program is to optimize the drift parameter of the geometric brownian motion for out-the-money options. The model of the underlying asset movement accounts for both continuous diffusion and discrete jumps in asset prices; however, the jump related parameters are set to zero in this program.

## Usage

```sh
$ git clone git@github.com:mlian031/importance-sampling.git
$ cd importance-sampling
$ git checkout eq-2.8-impl
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt
$ python main.py
```

## Background

### Importance sampling

In Monte Carlo simulation, importance sampling is a variance reduction technique based on changing the probability measure. The core idea is:
Given expectation $E[h(X)]$ where $X$ has density $f$, we can rewrite:

$$
\alpha = E[h(X)] = \int h(x)f(x)dx
$$

For any other density $g$ satisfying $f(x) > 0 \Rightarrow g(x) > 0$, we can represent this as:

$$
\alpha = \int h(x)\frac{f(x)}{g(x)}g(x)dx = \tilde{E}\left[h(X)\frac{f(X)}{g(X)}\right]
$$

where $\tilde{E}$ indicates expectation under $g$. This leads to the importance sampling estimator:

```math
\hat{\alpha}_g = \hat{\alpha}_g(n) = \frac{1}{n}\sum_{i=1}^n h(X_i)\frac{f(X_i)}{g(X_i)}
```

with $X_1,\ldots,X_n$ drawn from $g$. The ratio $f(X_i)/g(X_i)$ is called the likelihood ratio or Radon-Nikodym derivative.

> From Glasserman, Monte Carlo Methods in Financial Engineering

### A jump diffusion model

The Merton Jump Diffusion model extends geometric Brownian motion by adding a compound Poisson process to model sudden price jumps. The stochastic differential equation is:

$$
\frac{dS(t)}{S(t)} = \mu dt + \sigma dW(t) + dJ(t)
$$

where:

- $S(t)$ is the asset price at time $t$
- $\mu$ is the drift
- $\sigma$ is the volatility
- $W(t)$ is a standard Brownian motion
- $J(t)$ is a compound Poisson process given by:

$$
J(t) = \sum_{j=1}^{N(t)} (Y_j - 1)
$$

Here $N(t)$ is a Poisson process with intensity $\lambda$ and $Y_j$ are independent jump sizes, typically lognormally distributed:

$$
\log(Y_j) \sim N(a, b^2)
$$

The solution to the SDE is:

$$
S(t) = S(0)e^{(\mu-\frac{1}{2}\sigma^2)t+\sigma W(t)}\prod_{j=1}^{N(t)} Y_j
$$

For option pricing under the risk-neutral measure, the drift is adjusted to:

$$
\mu = r - \lambda m
$$

where $r$ is the risk-free rate and $m = E[Y_j - 1]$ compensates for the jumps.

> From Glasserman, Monte Carlo Methods in Financial Engineering

## Implementation

The program uses importance sampling to estimate the price of an out-the-money call option. The payoff function is:

$$
G(Z) = \max(S_T - K, 0)
$$

and the logarithmic transformation $F(Z)$ is: 

$$F(Z) = log(G(Z))$$ 

Since $G(Z) = 0$ whenever $S_T < K$, the logathrim is undefined in these cases. Consequently, $F(Z)$ is defined as a piecewise function:

$$
F(Z) = \begin{cases}
    log(S_T - K) & \text{if } S_T \geq K \\
    -\infty & \text{otherwise}
\end{cases}
$$

To avoid computational issues when $G(Z) = 0$, the program substitutes a small stabilization value ($10^{-6}$) in these cases.

The optimization problem follows Equation (2.8) from Glasserman and Heidelberger (2000), where the goal is to maximize:

$$
\max_{z \in D} \left\lbrace F(Z) - \frac{1}{2}z'z \right\rbrace
$$

Here, $z \in D$ represents the feasible set for the control variates or drift adjustment in the importance sampling distribution.

> From Glasserman, Asymptotically Optimal Importance Sampling and Stratified Sampling for Pricing Path-Dependent Options

## Notes on the implementation

### 1
Bounds for the optimization problem are set to $[-10^3, 10^3]$ for the drift parameter. The program encounters numerical issues when the lower bound is set lower than $-10^3$. I am not sure why this happens.

### 2 
There is a previous implementation using a fixed point method of iteratively searching for the optimal drift parameter. 

We begin with (2.5) from Glasserman and Heidelberger (2000):

$$
\min_{\mu} \mathbb{E} \left[ G(Z)^2 e^{-\mu' Z + (1/2) \mu' \mu} \mathbb{1}_D \right]
$$

Let 

$$
f(\mu) = \mathbb{E} \left[ G(Z)^2 e^{-\mu' Z + (1/2) \mu' \mu} \mathbb{1}_D \right] = \int_D G(z)^2 e^{-\mu z + \tfrac12 \mu^2} p(z) \mathrm{d}z
$$

Where $p(z)$ is the probability density function of Z and the indicator function returns 0 outside of $D$ so that integration only occurs on $D$. $D$ is the domain on which $Z$ (and hence $z$) takes values. 

We want $\frac{d}{d\mu} f(\mu)$.

$$
\frac{d}{d\mu} f(\mu) = \frac{d}{d\mu} \int_D G(z)^2 e^{-\mu z + \tfrac12 \mu^2} p(z) \mathrm{d}z = \int_D G(z)^2 \frac{d}{d\mu}\left[e^{-\mu z + \tfrac12 \mu^2}\right] \mathrm{d}z
$$

Now we differentiate:

$$
\frac{d}{d\mu} e^{-\mu z + \tfrac12 \mu^2} = e^{-\mu z + \tfrac12 \mu^2} \frac{d}{d\mu} \left[ -\mu z + \frac{1}{2} \mu^2 \right] \\
= e^{-\mu z + \tfrac12 \mu^2}  (-z + \mu)
$$

Putting this back into the integral, 

$$
\frac{d}{d\mu}f(\mu) = \int_D G(z)^2  {p}(z) \left[ e^{-\mu z + \tfrac12 \mu^2} (-Z + \mu) \right] \mathrm{d}z
$$

$$
\frac{d}{d\mu}f(\mu) = \mathbb{E}\left[ G(z)^2 e^{-\mu z + \tfrac12 \mu^2} (-Z + \mu) \mathbb{1}_D \right]
$$

To find the $\mu$ that minimizes $f(\mu)$, we set the derivative equal to zero:

$$
\mathbb{E}\left[ G(z)^2 e^{-\mu z + \tfrac12 \mu^2} (-Z + \mu) \mathbb{1}_D \right] = 0
$$


Define $w(Z,\mu) = G(Z)^2 e^{-\mu Z + \tfrac12 \mu^2} \mathbb{1}_D$

$$
\mathbb{E}\left[ w(Z,\mu)(-Z+\mu) \right] = 0 \\
$$
$$
\mathbb{E}\left[-Z w(Z,\mu) \right] + \mu\mathbb{E}\left[w(Z, \mu) \right] = 0 \\
$$
$$
-\mathbb{E}\left[Z w(Z, \mu)\right] + \mu\mathbb{E}\left[w(Z, \mu) \right] = 0 \\
$$
$$
\mu\mathbb{E}\left[w(Z, \mu) \right] = \mathbb{E}\left[Z w(Z, \mu)\right] \\
$$
$$
\mu = \frac{\mathbb{E}\left[Zw(Z, \mu)\right]}{\mathbb{E}\left[w(Z, \mu) \right]}
$$

Expressing it in terms of the definition of expected value,

$$
\mu = \frac{\sum_{i=1}^N Z_i G(Z_i)^2 e^{-\mu Z_i + \tfrac12\mu^2}}{\sum_{i=1}^N G(Z_i)^2 e^{-\mu Z_i + \tfrac12\mu^2}}
$$

This becomes our fixed-point equation.

Then we pick an initial guess $\mu^{(0)}$ and plug $\mu^{(0)}$ into the right-hand side to get $\mu^{(1)}$ and repeat until $| \mu^{k+1} - \mu^{k} | < 10^{-6}$ or the $\verb|numpy|$ tolerance value.

**Remark:** There is no significant difference in speed or accuracy between solving 2.8 or solving 2.5 using the fixed-point method. I will upload benchmarks soon.

# Sample

Parameters:
- $S_0 = 100$
- $K = 120$
- $r = 0.05$
- $\sigma = 0.2$
- $T = 1.0$

This is the result of an out-the-money option with $K = 120$ and $S_0 = 100$.


![](monte_carlo_vs_importance_sampling.png)

```
Path Count | MC Estimate | MC StdErr | IS Estimate | IS StdErr
    10000 |    3.242168 |   0.086589 |    3.245684 |   0.022313
    10531 |    3.251394 |   0.084733 |    3.250035 |   0.021726
    11090 |    3.253028 |   0.082399 |    3.251525 |   0.021171
    11679 |    3.251629 |   0.080240 |    3.245672 |   0.020642
    12299 |    3.243787 |   0.078044 |    3.250124 |   0.020122
    12952 |    3.243491 |   0.076026 |    3.246720 |   0.019591
    13640 |    3.250014 |   0.074358 |    3.249562 |   0.019104
    14364 |    3.249904 |   0.072350 |    3.245501 |   0.018619
    15127 |    3.253537 |   0.070675 |    3.246251 |   0.018142
    15931 |    3.240448 |   0.068642 |    3.246435 |   0.017676
    16777 |    3.238310 |   0.066796 |    3.246338 |   0.017232
    17668 |    3.245230 |   0.065231 |    3.248242 |   0.016785
    18606 |    3.245625 |   0.063629 |    3.244925 |   0.016363
    19594 |    3.248665 |   0.061904 |    3.249127 |   0.015933
    20635 |    3.248723 |   0.060376 |    3.248935 |   0.015523
    21730 |    3.248707 |   0.058847 |    3.247953 |   0.015128
    22884 |    3.253823 |   0.057354 |    3.249440 |   0.014746
    24100 |    3.245864 |   0.055853 |    3.244499 |   0.014376
    25380 |    3.247488 |   0.054517 |    3.250217 |   0.013999
    26727 |    3.251633 |   0.053032 |    3.247280 |   0.013646
    28147 |    3.249656 |   0.051683 |    3.248826 |   0.013296
    29642 |    3.241292 |   0.050248 |    3.247652 |   0.012957
    31216 |    3.249578 |   0.049142 |    3.248621 |   0.012622
    32874 |    3.246203 |   0.047805 |    3.248019 |   0.012300
    34619 |    3.240775 |   0.046580 |    3.246375 |   0.011992
    36458 |    3.236652 |   0.045344 |    3.245886 |   0.011686
    38394 |    3.245249 |   0.044189 |    3.247829 |   0.011385
    40433 |    3.255466 |   0.043184 |    3.247199 |   0.011093
    42580 |    3.248952 |   0.042036 |    3.247114 |   0.010810
    44842 |    3.246821 |   0.040956 |    3.248447 |   0.010531
    47223 |    3.251300 |   0.039942 |    3.246210 |   0.010268
    49731 |    3.248961 |   0.038920 |    3.247124 |   0.010006
    52372 |    3.246117 |   0.037881 |    3.246780 |   0.009750
    55153 |    3.251091 |   0.036968 |    3.247927 |   0.009498
    58082 |    3.243870 |   0.035985 |    3.247221 |   0.009256
    61166 |    3.248693 |   0.035093 |    3.247387 |   0.009020
    64415 |    3.249388 |   0.034195 |    3.246943 |   0.008791
    67836 |    3.253207 |   0.033350 |    3.246666 |   0.008567
    71438 |    3.246740 |   0.032448 |    3.246728 |   0.008348
    75232 |    3.247278 |   0.031593 |    3.246028 |   0.008136
    79227 |    3.250631 |   0.030838 |    3.247369 |   0.007926
    83435 |    3.244349 |   0.029996 |    3.248613 |   0.007722
    87865 |    3.245913 |   0.029237 |    3.246576 |   0.007527
    92532 |    3.248962 |   0.028517 |    3.247356 |   0.007334
    97446 |    3.243678 |   0.027754 |    3.248291 |   0.007146
   102620 |    3.249547 |   0.027063 |    3.246106 |   0.006965
   108070 |    3.249107 |   0.026380 |    3.246501 |   0.006787
   113809 |    3.243742 |   0.025696 |    3.247768 |   0.006613
   119853 |    3.244529 |   0.025041 |    3.247166 |   0.006444
   126218 |    3.251543 |   0.024427 |    3.247347 |   0.006280
   132921 |    3.248513 |   0.023780 |    3.248002 |   0.006118
   139980 |    3.248647 |   0.023185 |    3.247864 |   0.005962
   147414 |    3.249041 |   0.022585 |    3.246277 |   0.005811
   155242 |    3.246757 |   0.022008 |    3.248360 |   0.005662
   163486 |    3.247470 |   0.021442 |    3.246496 |   0.005519
   172169 |    3.248384 |   0.020907 |    3.247124 |   0.005377
   181312 |    3.249833 |   0.020378 |    3.248417 |   0.005239
   190940 |    3.248565 |   0.019851 |    3.247094 |   0.005106
   201080 |    3.246442 |   0.019340 |    3.247461 |   0.004975
   211759 |    3.245319 |   0.018828 |    3.247678 |   0.004848
   223005 |    3.248796 |   0.018365 |    3.247548 |   0.004724
   234847 |    3.246237 |   0.017893 |    3.247491 |   0.004603
   247319 |    3.246046 |   0.017437 |    3.247357 |   0.004486
   260453 |    3.247491 |   0.016988 |    3.247228 |   0.004371
   274285 |    3.248375 |   0.016557 |    3.246649 |   0.004260
   288851 |    3.246744 |   0.016133 |    3.247400 |   0.004151
   304190 |    3.247346 |   0.015721 |    3.247240 |   0.004045
   320345 |    3.247192 |   0.015323 |    3.247318 |   0.003942
   337357 |    3.247956 |   0.014926 |    3.246831 |   0.003841
   355272 |    3.244643 |   0.014544 |    3.247776 |   0.003743
   374139 |    3.246733 |   0.014175 |    3.247684 |   0.003647
   394008 |    3.246905 |   0.013812 |    3.247372 |   0.003554
   414932 |    3.248122 |   0.013468 |    3.247339 |   0.003463
   436967 |    3.246006 |   0.013118 |    3.247476 |   0.003375
   460173 |    3.246442 |   0.012782 |    3.247814 |   0.003289
   484610 |    3.248323 |   0.012458 |    3.247083 |   0.003205
   510346 |    3.246491 |   0.012135 |    3.247278 |   0.003123
   537448 |    3.250577 |   0.011838 |    3.247011 |   0.003043
   565990 |    3.246410 |   0.011525 |    3.247438 |   0.002965
   596047 |    3.247892 |   0.011237 |    3.247913 |   0.002890
   627700 |    3.248072 |   0.010944 |    3.247027 |   0.002816
   661035 |    3.247231 |   0.010669 |    3.247456 |   0.002744
   696140 |    3.245895 |   0.010392 |    3.247751 |   0.002674
   733108 |    3.247932 |   0.010131 |    3.247437 |   0.002606
   772041 |    3.247752 |   0.009869 |    3.247739 |   0.002539
   813040 |    3.245060 |   0.009612 |    3.247311 |   0.002474
   856217 |    3.249692 |   0.009376 |    3.247328 |   0.002411
   901687 |    3.247719 |   0.009131 |    3.247000 |   0.002349
   949572 |    3.247761 |   0.008899 |    3.247277 |   0.002289
  1000000 |    3.246826 |   0.008671 |    3.247497 |   0.002231

Variance Reduction Multiple: 15.11
```

## Citations

P. Glasserman, Monte Carlo Methods in Financial Engineering, vol. 53. New York, NY: Springer New York, 2003. doi: 10.1007/978-0-387-21617-1.

Glasserman, P., Heidelberger, P. and Shahabuddin, P. (1999), Asymptotically Optimal Importance Sampling and Stratification for Pricing Path-Dependent Options. Mathematical Finance, 9: 117-152. https://doi.org/10.1111/1467-9965.00065
