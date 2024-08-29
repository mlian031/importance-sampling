use rand::distributions::Distribution;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::f64;
/**
*
* Name: Mike
* Date: Aug 28
* Description:
* Demonstration of importance sampling using GBM
*
*
*/

struct EuroCallOption {
    initial: f64,
    mu: f64,
    sigma: f64,
    num_sims: usize,
    strike: f64,
    time: f64,
    time_steps: usize,
    risk_free_rate: f64,
}

fn bs_call_option_price(params: &EuroCallOption) -> f64 {
    let d_1 = ((params.initial.ln() - params.strike.ln())
        + (params.risk_free_rate + 0.5 * params.sigma.powi(2)) * params.time)
        / (params.sigma * params.time.sqrt());
    let d_2 = d_1 - (params.sigma * params.time);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let normal_d1 = normal.cdf(d_1);
    let normal_d2 = normal.cdf(d_2);

    (normal_d1 * params.initial)
        - (normal_d2 * params.strike * f64::exp(-1.0 * params.risk_free_rate * params.time))
}

fn main() {
    let euro_call_option = EuroCallOption {
        initial: 100.0,
        mu: 0.05,
        sigma: 0.10,
        time: 1.0,
        time_steps: 252,
        num_sims: 10000,
        strike: 150.0,
        risk_free_rate: 0.05,
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    let mut mc_expected_payoffs = vec![0.0; euro_call_option.num_sims];
    let mut mc_is_expected_payoffs = vec![0.0; euro_call_option.num_sims];

    for i in 0..euro_call_option.num_sims {
        let z = normal.sample(&mut rng);
        let drift =
            (euro_call_option.mu - 0.5 * euro_call_option.sigma.powi(2)) * euro_call_option.time;
        let diffusion = euro_call_option.sigma * euro_call_option.time.sqrt();
        let s0 = euro_call_option.initial;

        let z_t = drift + diffusion * z;
        mc_expected_payoffs[i] =
            f64::exp(-1.0 * euro_call_option.risk_free_rate * euro_call_option.time)
                * (s0 * f64::exp(z_t) - euro_call_option.strike).max(0.0);
    }

    // Importance sampling
    for i in 0..euro_call_option.num_sims {
        let z = normal.sample(&mut rng);
        let s0 = euro_call_option.initial;
        let drift = (f64::ln(euro_call_option.strike / s0) - 0.5 * euro_call_option.sigma.powi(2))
            * euro_call_option.time;
        let diffusion = euro_call_option.sigma * euro_call_option.time.sqrt();

        let z_t = drift + diffusion * z;
        let s_t = s0 * f64::exp(z_t);
        let old_prob_measure = Normal::new(
            (euro_call_option.mu - 0.5 * euro_call_option.sigma.powi(2)) * euro_call_option.time,
            euro_call_option.sigma * euro_call_option.time.sqrt(),
        )
        .unwrap();
        let new_prob_measure = Normal::new(
            (f64::ln(euro_call_option.strike / s0))
                - 0.5 * euro_call_option.sigma.powi(2) * euro_call_option.time,
            euro_call_option.sigma * euro_call_option.time.sqrt(),
        )
        .unwrap();

        let adjustment_factor = old_prob_measure.pdf(z_t) / new_prob_measure.pdf(z_t);
        mc_is_expected_payoffs[i] =
            f64::exp(-1.0 * euro_call_option.risk_free_rate * euro_call_option.time)
                * (s_t - euro_call_option.strike).max(0.0)
                * adjustment_factor;
    }

    let mean_mc_payoff: f64 =
        mc_expected_payoffs.iter().sum::<f64>() / euro_call_option.num_sims as f64;
    let mean_mc_is_payoff: f64 =
        mc_is_expected_payoffs.iter().sum::<f64>() / euro_call_option.num_sims as f64;

    let variance_mc_payoff = mc_expected_payoffs
        .iter()
        .map(|x| (x - mean_mc_payoff).powi(2))
        .sum::<f64>()
        / (euro_call_option.num_sims - 1) as f64;

    let variance_mc_is_payoff = mc_is_expected_payoffs
        .iter()
        .map(|x| (x - mean_mc_is_payoff).powi(2))
        .sum::<f64>()
        / (euro_call_option.num_sims - 1) as f64;

    let efficiency = variance_mc_payoff / variance_mc_is_payoff;

    let black_scholes_estimate = bs_call_option_price(&euro_call_option);

    println!("Simple Monte Carlo Estimate: {}", mean_mc_payoff);
    println!("Importance Sampling Estimate: {}", mean_mc_is_payoff);
    println!("Black Scholes Estimate: {}", black_scholes_estimate);

    println!("Efficiency Gain: {}", efficiency);
}
