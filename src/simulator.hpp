#pragma once
#include <vector>
#include <chrono>
#include "gbm.hpp"
#include "stats.hpp"

struct SimResult {
    std::vector<double> final_prices;
    double elapsed_ms;
};

// Run num_paths GBM simulations in parallel using OpenMP.
//
// Each thread gets its own Xoshiro256pp seeded with (base_seed + thread_id)
// so there is zero shared mutable state on the hot path — no mutexes,
// no atomics, no false sharing. This is the standard pattern for
// parallel Monte Carlo in production systems.
SimResult run_simulation(const GBMParams& params, int num_paths, uint64_t base_seed = 42) {
    std::vector<double> prices;
    prices.resize(num_paths);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // Antithetic variates: each iteration produces 2 prices, so we only
    // need num_paths/2 RNG calls. Pre-size to exact count.
    const int half = num_paths / 2;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < half; ++i) {
        // Thread-local RNG: seed offset by loop index ensures independent streams.
        Xoshiro256pp rng(base_seed + static_cast<uint64_t>(i) * 2654435761ULL);
        auto [p_pos, p_neg] = simulate_antithetic(params, rng);
        prices[2 * i]     = p_pos;
        prices[2 * i + 1] = p_neg;
    }
    // Handle the odd path if num_paths is not even.
    if (num_paths % 2 != 0) {
        Xoshiro256pp rng(base_seed + static_cast<uint64_t>(half) * 2654435761ULL + 1);
        prices[num_paths - 1] = simulate_path(params, rng);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    return SimResult{std::move(prices), elapsed_ms};
}
