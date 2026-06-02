#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct RiskReport {
    double mean;
    double median;
    double std_dev;
    double p5;          // 5th percentile final price
    double p10;
    double p25;
    double p75;
    double p90;
    double p95;
    double var_95;      // Value at Risk at 95% confidence (loss from S0)
    double var_99;
    double cvar_95;     // Conditional VaR (Expected Shortfall) at 95%
    double prob_profit; // P(final price > S0)
    double S0;
};

// All inputs are final simulated prices. The vector is sorted in-place.
inline RiskReport compute_risk(std::vector<double>& prices, double S0) {
    std::sort(prices.begin(), prices.end());
    const size_t n = prices.size();

    auto percentile = [&](double p) -> double {
        const double idx = p * (n - 1);
        const size_t lo  = static_cast<size_t>(idx);
        const size_t hi  = lo + 1 < n ? lo + 1 : lo;
        const double frac = idx - lo;
        return prices[lo] * (1.0 - frac) + prices[hi] * frac;
    };

    const double sum  = std::accumulate(prices.begin(), prices.end(), 0.0);
    const double mean = sum / n;

    double sq_sum = 0.0;
    for (double p : prices) sq_sum += (p - mean) * (p - mean);
    const double std_dev = std::sqrt(sq_sum / n);

    // VaR: worst loss at given confidence level.
    // E.g. VaR_95 = S0 - 5th percentile price (5% of paths end below this).
    const double p5  = percentile(0.05);
    const double p1  = percentile(0.01);
    const double var_95 = S0 - p5;
    const double var_99 = S0 - p1;

    // CVaR / Expected Shortfall: average loss in the worst 5% of paths.
    const size_t tail = static_cast<size_t>(0.05 * n);
    double tail_sum = 0.0;
    for (size_t i = 0; i < tail; ++i) tail_sum += prices[i];
    const double cvar_95 = S0 - (tail > 0 ? tail_sum / tail : p5);

    // Probability of profit: fraction of paths above initial price.
    const auto it = std::lower_bound(prices.begin(), prices.end(), S0);
    const double prob_profit = 1.0 - static_cast<double>(it - prices.begin()) / n;

    return RiskReport{
        mean,
        percentile(0.50),
        std_dev,
        p5,
        percentile(0.10),
        percentile(0.25),
        percentile(0.75),
        percentile(0.90),
        percentile(0.95),
        var_95,
        var_99,
        cvar_95,
        prob_profit,
        S0
    };
}
