#pragma once
#include <cmath>
#include <utility>
#include "rng.hpp"

// Merton (1976) Jump-Diffusion model policy.
//
// Extends GBM with a compound Poisson jump process:
//   dS = S*(mu*dt + sigma*dW) + S*(J-1)*dN
//
// where:
//   dN ~ Poisson(lambda*dt)  — jump arrival (Bernoulli approx for small dt)
//   J  = exp(mu_j + sigma_j * Z_j),  Z_j ~ N(0,1)  — log-normal jump size
//
// Drift correction:
//   The jumps add expected return lambda*kappa per unit time, where
//   kappa = E[J-1] = exp(mu_j + 0.5*sigma_j^2) - 1.
//   To keep the total expected return equal to mu, we subtract lambda*kappa
//   from the continuous drift term. Without this the model overestimates
//   expected return.
//
// Why this matters vs GBM:
//   GBM produces log-normal terminal prices — thin tails, no discontinuities.
//   Jump-diffusion produces fat tails and discrete price gaps, matching
//   empirical return distributions far better (excess kurtosis > 0).
//   The result is a materially higher VaR and CVaR for the same mu/sigma.

struct MertonJump {
    struct Params {
        double S0;
        double mu;       // annualised drift (total expected return)
        double sigma;    // annualised diffusion volatility
        double T;
        int    steps;
        double lambda;   // jump intensity: expected jumps per year (e.g. 3.0)
        double mu_j;     // mean log-jump size (e.g. -0.05 for avg -5% crash)
        double sigma_j;  // std dev of log-jump size (e.g. 0.08)
    };

    static const char* name() { return "Merton Jump-Diffusion"; }

    static double simulate_path(const Params& p, Xoshiro256pp& rng) {
        const double dt         = p.T / p.steps;
        const double kappa      = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
        const double drift      = (p.mu - 0.5 * p.sigma * p.sigma - p.lambda * kappa) * dt;
        const double diffuse    = p.sigma * std::sqrt(dt);
        const double jump_prob  = p.lambda * dt;

        double S = p.S0;
        for (int i = 0; i < p.steps; ++i) {
            S *= std::exp(drift + diffuse * rng.nextNormal());
            // Poisson(lambda*dt) ≈ Bernoulli(lambda*dt) for small lambda*dt.
            // For lambda=3, dt=1/252: P(jump) ≈ 0.012 per step.
            if (rng.nextDouble() < jump_prob)
                S *= std::exp(p.mu_j + p.sigma_j * rng.nextNormal());
        }
        return S;
    }

    static std::pair<double, double> simulate_antithetic(const Params& p, Xoshiro256pp& rng) {
        const double dt         = p.T / p.steps;
        const double kappa      = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
        const double drift      = (p.mu - 0.5 * p.sigma * p.sigma - p.lambda * kappa) * dt;
        const double diffuse    = p.sigma * std::sqrt(dt);
        const double jump_prob  = p.lambda * dt;

        double S_pos = p.S0, S_neg = p.S0;
        for (int i = 0; i < p.steps; ++i) {
            const double Z      = rng.nextNormal();
            const double shock  = diffuse * Z;
            S_pos *= std::exp(drift + shock);
            S_neg *= std::exp(drift - shock);
            // Jumps are not antithetic — they represent discrete events
            // (crashes) that don't have a natural paired opposite.
            // Both paths share the same jump arrival draw for consistency.
            if (rng.nextDouble() < jump_prob) {
                const double log_jump = p.mu_j + p.sigma_j * rng.nextNormal();
                S_pos *= std::exp(log_jump);
                S_neg *= std::exp(log_jump);
            }
        }
        return {S_pos, S_neg};
    }
};
