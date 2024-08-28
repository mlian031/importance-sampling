/*
*
* Name: Mike
* Date: Aug 22
* Description:
* Demonstration of importance sampling using GBM
*
*/

use std::f64;

use rand::distributions::Distribution;
use statrs::distribution::{ContinuousCDF, Normal};

struct EuroCallOption {
    initial: f64,
    mu: f64,
    sigma: f64,
    time: f64,
    time_steps: usize,
    num_sims: usize,
    strike: f64,
    risk_free_rate: f64,
}

fn simulate_terminal_price(params: &EuroCallOption) -> f64 {
    let dt = params.time / params.time_steps as f64;
    let mut rng = rand::thread_rng();

    let normal = Normal::new(0.0, 1.0).unwrap();

    let drift = (params.mu - 0.5 * params.sigma.powi(2)) * dt;
    let diffusion = params.sigma * dt.sqrt();

    let mut s_t = params.initial;

    for _ in 0..params.time_steps {
        let std_normal_z = normal.sample(&mut rng);
        s_t *= f64::exp(drift + diffusion * std_normal_z);
    }

    s_t
}

fn mc_expected_payoff(params: &EuroCallOption) -> f64 {
    let mut sum_payoff = 0.0;

    for _ in 0..params.num_sims {
        let final_price = simulate_terminal_price(params);
        let payoff = (final_price - params.strike).max(0.0);
        sum_payoff += payoff;
    }

    (sum_payoff / params.num_sims as f64) * f64::exp(-1.0 * params.risk_free_rate * params.time)
}

fn bs_expected_payoff(params: &EuroCallOption) -> f64 {
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
    let params = EuroCallOption {
        initial: 143.23,
        mu: 0.05,
        sigma: 0.2,
        time: 1.0,
        time_steps: 252,
        num_sims: 5000,
        strike: 158.23,
        risk_free_rate: 0.05,
    };

    println!("Running {} simulations", params.num_sims);

    let expected_payoff = mc_expected_payoff(&params);
    println!("Estimated Expected Payoff: {}", expected_payoff);

    let bs_expected_payoff = bs_expected_payoff(&params);
    println!("Black Scholes Calculated Payoff: {}", bs_expected_payoff);
}
