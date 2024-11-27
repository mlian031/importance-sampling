# Importance Sampling

The goal of this program is to optimize the drift parameter of the geometric brownian motion for out-the-money options. The model of the underlying asset movement accounts for both continuous diffusion and discrete jumps in asset prices; however, the jump related parameters are set to zero in this program.

## Usage

```sh
$ git clone git@github.com:mlian031/importance-sampling.git
```

To use the importance sampling module, checkout to `gbm-importance-sampling`

```sh
$ git checkout gbm-importance-sampling
$ pip install -r requirements.txt
$ python main.py
```

## Citations

P. Glasserman, Monte Carlo Methods in Financial Engineering, vol. 53. New York, NY: Springer New York, 2003. doi: 10.1007/978-0-387-21617-1.

Q. Zhao, G. Liu and G. Gu, "Variance Reduction Techniques of Importance Sampling Monte Carlo Methods for Pricing Options," Journal of Mathematical Finance, Vol. 3 No. 4, 2013, pp. 431-436. doi: 10.4236/jmf.2013.34045.

Y. Su and M. C. Fu, "Importance sampling in derivative securities pricing," 2000 Winter Simulation Conference Proceedings (Cat. No.00CH37165), Orlando, FL, USA, 2000, pp. 587-596 vol.1, doi: 10.1109/WSC.2000.899767. keywords: {Monte Carlo methods;Pricing;Security;Stochastic processes;Educational institutions;Analysis of variance;Approximation algorithms;Computational modeling;Sampling methods;Cost accounting},
