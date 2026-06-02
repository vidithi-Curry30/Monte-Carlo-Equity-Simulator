#pragma once
#include <cmath>
#include <vector>
#include "rng.hpp"

// Geometric Brownian Motion: the continuous-time model underlying Black-Scholes.
//
// Price dynamics:  dS = S * (mu * dt + sigma * sqrt(dt) * Z),  Z ~ N(0,1)
// Exact discrete solution (no Euler approximation error):
//   S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
//
// The 0.5*sigma^2 (Ito correction) converts the arithmetic drift to the
// geometric drift required for the log-normal distribution.

struct GBMParams {
    double S0;       // initial price
    double mu;       // annualised drift (e.g. 0.08 = 8%)
    double sigma;    // annualised volatility (e.g. 0.20 = 20%)
    double T;        // horizon in years (e.g. 0.25 = 3 months)
    int    steps;    // number of time steps
};

// Simulate a single GBM path. Returns final price only (saves memory vs
// storing every step — the hot path in Monte Carlo only needs terminal values).
inline double simulate_path(const GBMParams& p, Xoshiro256pp& rng) {
    const double dt      = p.T / p.steps;
    const double drift   = (p.mu - 0.5 * p.sigma * p.sigma) * dt;
    const double diffuse = p.sigma * std::sqrt(dt);

    double S = p.S0;
    for (int i = 0; i < p.steps; ++i) {
        S *= std::exp(drift + diffuse * rng.nextNormal());
    }
    return S;
}

// Antithetic variates variance reduction:
// For each normal Z, also simulate with -Z. The two paths are negatively
// correlated, so averaging them cancels the first-order Monte Carlo error.
// Result: same accuracy as 2x the paths at almost no extra cost.
inline std::pair<double, double> simulate_antithetic(const GBMParams& p, Xoshiro256pp& rng) {
    const double dt      = p.T / p.steps;
    const double drift   = (p.mu - 0.5 * p.sigma * p.sigma) * dt;
    const double diffuse = p.sigma * std::sqrt(dt);

    double S_pos = p.S0;
    double S_neg = p.S0;
    for (int i = 0; i < p.steps; ++i) {
        const double Z = rng.nextNormal();
        const double shock = diffuse * Z;
        S_pos *= std::exp(drift + shock);
        S_neg *= std::exp(drift - shock);
    }
    return {S_pos, S_neg};
}
