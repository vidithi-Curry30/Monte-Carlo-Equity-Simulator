#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include "../src/gbm.hpp"
#include "../src/jump_diffusion.hpp"
#include "../src/simulator.hpp"
#include "../src/stats.hpp"

// ── False sharing demonstration ───────────────────────────────────────────
//
// False sharing occurs when two threads write to different variables that
// happen to occupy the same cache line (typically 64 bytes on x86).
// The CPU cache coherence protocol (MESI) forces the entire line to be
// invalidated and transferred between cores on every write — even though
// the threads are touching logically independent data.
//
// This benchmark makes the effect concrete and measurable.

static constexpr int    FS_THREADS    = 4;
static constexpr long   FS_ITERATIONS = 50'000'000L;

// Bad: all counters packed into 8 bytes each — up to 8 per cache line.
struct BadCounter {
    long value = 0;
};

// Good: each counter owns an entire cache line.
struct alignas(64) GoodCounter {
    long value = 0;
};

template<typename Counter>
double bench_false_sharing(const char* label) {
    std::vector<Counter> counters(FS_THREADS);
    std::vector<std::thread> threads;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < FS_THREADS; ++t) {
        threads.emplace_back([&counters, t] {
            for (long i = 0; i < FS_ITERATIONS; ++i)
                counters[t].value++;
        });
    }
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  " << std::left << std::setw(30) << label
              << std::right << std::setw(8) << std::fixed << std::setprecision(1)
              << ms << " ms\n";
    return ms;
}

// ── Model comparison ──────────────────────────────────────────────────────

template<typename Model>
void bench_model(const typename Model::Params& params, int paths, int threads, uint64_t seed) {
    auto result = run_simulation<Model>(params, paths, threads, seed);
    auto report = compute_risk(result.final_prices, params.S0);
    const double throughput = paths / (result.elapsed_ms / 1000.0) / 1e6;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::left << std::setw(26) << Model::name()
              << " | " << std::right << std::setw(7) << result.elapsed_ms << " ms"
              << " | " << std::setprecision(1) << std::setw(5) << throughput << "M p/s"
              << " | VaR95=$" << std::setprecision(2) << report.var_95
              << "  CVaR95=$" << report.cvar_95
              << "\n";
}

int main() {
    const int N_THREADS = static_cast<int>(std::thread::hardware_concurrency());
    const int PATHS     = 1'000'000;

    // ── 1. False sharing ──────────────────────────────────────────────────
    std::cout << "=== False Sharing Benchmark (" << FS_THREADS << " threads, "
              << FS_ITERATIONS / 1'000'000 << "M increments each) ===\n";
    const double t_bad  = bench_false_sharing<BadCounter> ("Unaligned (false sharing)");
    const double t_good = bench_false_sharing<GoodCounter>("alignas(64) (no sharing)");
    std::cout << "  Speedup: " << std::setprecision(1) << (t_bad / t_good) << "x\n";
    std::cout << "  sizeof(BadCounter)="  << sizeof(BadCounter)
              << "  sizeof(GoodCounter)=" << sizeof(GoodCounter) << "\n\n";

    // ── 2. Model comparison: GBM vs Jump-Diffusion ────────────────────────
    const double S0 = 150.0, mu = 0.08, sigma = 0.20, T = 0.25;
    std::cout << "=== Model Comparison (" << PATHS/1'000'000 << "M paths, "
              << N_THREADS << " threads) ===\n";
    std::cout << "  S0=$" << S0 << "  mu=" << mu*100 << "%"
              << "  sigma=" << sigma*100 << "%  T=" << T << "yr\n\n";

    GBM::Params gbm_p{S0, mu, sigma, T, 252};
    bench_model<GBM>(gbm_p, PATHS, N_THREADS, 42);

    MertonJump::Params jump_p{S0, mu, sigma, T, 252, 3.0, -0.05, 0.08};
    bench_model<MertonJump>(jump_p, PATHS, N_THREADS, 42);

    std::cout << "\n  Jump-diffusion produces higher VaR/CVaR because it models\n"
              << "  discrete price gaps (earnings, macro shocks) that GBM cannot.\n\n";

    // ── 3. Thread scaling ─────────────────────────────────────────────────
    std::cout << "=== Thread Scaling (GBM, " << PATHS/1'000'000 << "M paths) ===\n";
    double t_single = 0;
    for (int t = 1; t <= N_THREADS; t = (t == 1 ? 2 : t + 2)) {
        auto result = run_simulation<GBM>(gbm_p, PATHS, t, 42);
        const double tput = PATHS / (result.elapsed_ms / 1000.0) / 1e6;
        if (t == 1) t_single = result.elapsed_ms;
        std::cout << "  threads=" << std::setw(2) << t
                  << "  " << std::setw(7) << std::fixed << std::setprecision(1)
                  << result.elapsed_ms << " ms"
                  << "  (" << std::setprecision(1) << tput << "M p/s)"
                  << "  speedup=" << std::setprecision(2) << (t_single / result.elapsed_ms) << "x"
                  << "\n";
    }

    return 0;
}
