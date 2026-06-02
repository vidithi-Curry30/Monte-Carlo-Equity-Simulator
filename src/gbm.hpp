#pragma once
#include <cmath>
#include <utility>
#include "rng.hpp"

// Geometric Brownian Motion model policy.
//
// Satisfies the Model concept required by Simulator<Model>:
//   - nested Params struct
//   - static simulate_path(Params, rng)  -> double
//   - static simulate_antithetic(Params, rng) -> pair<double,double>
//
// GBM exact discrete solution (no Euler error):
//   S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
//
// The 0.5*sigma^2 (Ito correction) converts arithmetic drift to geometric
// drift so that E[S(T)] = S0 * exp(mu*T) as required.

struct GBM {
    struct Params {
        double S0;
        double mu;     // annualised drift
        double sigma;  // annualised volatility
        double T;      // horizon in years
        int    steps;
    };

    static const char* name() { return "GBM"; }

    static double simulate_path(const Params& p, Xoshiro256pp& rng) {
        const double dt      = p.T / p.steps;
        const double drift   = (p.mu - 0.5 * p.sigma * p.sigma) * dt;
        const double diffuse = p.sigma * std::sqrt(dt);
        double S = p.S0;
        for (int i = 0; i < p.steps; ++i)
            S *= std::exp(drift + diffuse * rng.nextNormal());
        return S;
    }

    // Antithetic variates: simulate paired paths with +Z and -Z.
    // The negative correlation between pairs halves estimator variance
    // at almost no extra cost vs running two independent paths.
    static std::pair<double, double> simulate_antithetic(const Params& p, Xoshiro256pp& rng) {
        const double dt      = p.T / p.steps;
        const double drift   = (p.mu - 0.5 * p.sigma * p.sigma) * dt;
        const double diffuse = p.sigma * std::sqrt(dt);
        double S_pos = p.S0, S_neg = p.S0;
        for (int i = 0; i < p.steps; ++i) {
            const double shock = diffuse * rng.nextNormal();
            S_pos *= std::exp(drift + shock);
            S_neg *= std::exp(drift - shock);
        }
        return {S_pos, S_neg};
    }
};
