#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../src/gbm.hpp"
#include "../src/simulator.hpp"

// Measures raw simulation throughput and the speedup from antithetic variates
// and OpenMP parallelism. Run this to produce the numbers for your README.

static double bench_run(const GBMParams& p, int paths, bool antithetic, uint64_t seed) {
    std::vector<double> prices(paths);
    const int half = paths / 2;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (antithetic) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < half; ++i) {
            Xoshiro256pp rng(seed + static_cast<uint64_t>(i) * 2654435761ULL);
            auto [a, b] = simulate_antithetic(p, rng);
            prices[2*i] = a; prices[2*i+1] = b;
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < paths; ++i) {
            Xoshiro256pp rng(seed + static_cast<uint64_t>(i) * 2654435761ULL);
            prices[i] = simulate_path(p, rng);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const GBMParams params{150.0, 0.08, 0.20, 1.0, 252};
    const int PATHS = 2'000'000;
    const int REPS  = 5;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Benchmark: " << PATHS/1'000'000 << "M paths, "
              << params.steps << " steps each, " << REPS << " repetitions\n\n";

    auto run_avg = [&](bool antithetic) {
        double total = 0;
        for (int r = 0; r < REPS; ++r)
            total += bench_run(params, PATHS, antithetic, 42 + r);
        return total / REPS;
    };

    const double t_plain      = run_avg(false);
    const double t_antithetic = run_avg(true);

    const double mp_plain      = PATHS / (t_plain      / 1000.0) / 1e6;
    const double mp_antithetic = PATHS / (t_antithetic / 1000.0) / 1e6;

    std::cout << "  Standard MC     : " << std::setw(8) << t_plain
              << " ms avg  (" << mp_plain << " M paths/sec)\n";
    std::cout << "  Antithetic var. : " << std::setw(8) << t_antithetic
              << " ms avg  (" << mp_antithetic << " M paths/sec)\n";
    std::cout << "\n  Antithetic speedup: " << (t_plain / t_antithetic) << "x\n";

#ifdef _OPENMP
    int n_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        n_threads = omp_get_num_threads();
    }
    std::cout << "  OpenMP threads: " << n_threads << "\n";
#else
    std::cout << "  (Built without OpenMP — single-threaded)\n";
#endif

    std::cout << "\nRe-run with OMP_NUM_THREADS=1 to measure single-thread baseline.\n";
    return 0;
}
