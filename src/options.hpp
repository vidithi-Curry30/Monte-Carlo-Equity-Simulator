#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "simulator.hpp"
#include "stats.hpp"

struct OptionResult {
    double mc_price;    // Monte Carlo option price
    double bs_price;    // Black-Scholes closed-form price (GBM only; 0 for other models)
    double bs_error;    // mc_price - bs_price (should be within ~2x std_error for GBM)
    double std_error;   // MC standard error: stddev(payoffs) / sqrt(N)
    double delta;       // finite-difference delta
    double gamma;       // finite-difference gamma
    double vega;        // finite-difference vega (per 1% vol move)
};

// Normal CDF via erfc. Accurate to ~1e-15 across the real line.
static inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Black-Scholes closed-form price for a European call.
// Used to validate the GBM simulation — they should agree within MC error.
double black_scholes_call(double S0, double K, double r, double T, double sigma) {
    if (T <= 0 || sigma <= 0 || S0 <= 0 || K <= 0) return 0.0;
    const double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T)
                      / (sigma * std::sqrt(T));
    const double d2 = d1 - sigma * std::sqrt(T);
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// Compute MC call price and standard error from a set of terminal prices.
// Separated so we can reuse terminal prices already simulated for risk.
static std::pair<double, double> mc_call_price(
    const std::vector<double>& terminal_prices,
    double K, double r, double T)
{
    const size_t n = terminal_prices.size();
    std::vector<double> payoffs(n);
    for (size_t i = 0; i < n; ++i)
        payoffs[i] = std::max(terminal_prices[i] - K, 0.0);

    const double mean = [&] {
        double s = 0; for (double p : payoffs) s += p; return s / n;
    }();
    const double var = [&] {
        double s = 0;
        for (double p : payoffs) s += (p - mean) * (p - mean);
        return s / (n - 1);
    }();

    const double price    = std::exp(-r * T) * mean;
    const double std_err  = std::exp(-r * T) * std::sqrt(var / n);
    return {price, std_err};
}

// Price a European call and compute finite-difference Greeks.
//
// Greeks use central finite differences with a 1% bump in S0 (delta/gamma)
// and a 1 vol-point bump in sigma (vega). The same RNG seed is used for
// all three runs — "common random numbers" — so the paths differ only due
// to the parameter change, not noise. This dramatically reduces the variance
// of the finite-difference estimate vs using independent seeds.
//
// Step size choice: 1% of S0 is a practical compromise. Too small and
// floating-point cancellation dominates; too large and the finite difference
// approximates a chord rather than a derivative. For a $150 stock, ε=$1.50
// keeps both errors small.
template<typename Model>
OptionResult price_call(
    typename Model::Params params,  // copy — we'll modify S0/sigma for bumps
    double   K,
    double   r,
    int      n_paths,
    int      n_threads,
    uint64_t seed = 42)
{
    // Risk-neutral pricing: drift must equal the risk-free rate r, not the
    // historical (physical measure) drift mu. Black-Scholes makes the same
    // substitution when deriving its closed form. Using mu here would
    // produce MC prices that diverge from Black-Scholes by the equity
    // risk premium — a real economic quantity, not a modelling error,
    // but the wrong answer for no-arbitrage option pricing.
    params.mu = r;

    const double S0    = params.S0;
    const double sigma = params.sigma;
    const double T     = params.T;
    const double eps_s = 0.01 * S0;    // 1% of spot for delta/gamma
    const double eps_v = 0.01;         // 1 vol point for vega (absolute)

    // Base simulation (also used for risk report in main)
    auto base = run_simulation<Model>(params, n_paths, n_threads, seed);
    auto [price, std_err] = mc_call_price(base.final_prices, K, r, T);

    // Up/down bumps for delta and gamma — same seed = common random numbers
    params.S0 = S0 + eps_s;
    auto up_s = run_simulation<Model>(params, n_paths, n_threads, seed);
    auto [price_up_s, _1] = mc_call_price(up_s.final_prices, K, r, T);

    params.S0 = S0 - eps_s;
    auto dn_s = run_simulation<Model>(params, n_paths, n_threads, seed);
    auto [price_dn_s, _2] = mc_call_price(dn_s.final_prices, K, r, T);

    params.S0    = S0;  // restore
    params.sigma = sigma + eps_v;
    auto up_v = run_simulation<Model>(params, n_paths, n_threads, seed);
    auto [price_up_v, _3] = mc_call_price(up_v.final_prices, K, r, T);

    params.sigma = sigma - eps_v;
    auto dn_v = run_simulation<Model>(params, n_paths, n_threads, seed);
    auto [price_dn_v, _4] = mc_call_price(dn_v.final_prices, K, r, T);

    const double delta = (price_up_s - price_dn_s) / (2.0 * eps_s);
    const double gamma = (price_up_s - 2.0 * price + price_dn_s) / (eps_s * eps_s);
    const double vega  = (price_up_v - price_dn_v) / (2.0 * eps_v);  // per 1% vol

    const double bs = black_scholes_call(S0, K, r, T, sigma);

    return OptionResult{price, bs, price - bs, std_err, delta, gamma, vega};
}
