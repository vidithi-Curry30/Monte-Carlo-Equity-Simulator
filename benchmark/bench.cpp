#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include "../src/gbm.hpp"
#include "../src/jump_diffusion.hpp"
#include "../src/simulator.hpp"
#include "../src/spsc_queue.hpp"
#include "../src/stats.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// 1. False sharing
// ─────────────────────────────────────────────────────────────────────────────
//
// Each thread increments its own counter in a tight loop. With unaligned
// counters, adjacent threads share a cache line and the MESI protocol forces
// every write to invalidate other cores' copies — even though they're
// writing different memory locations.
//
// A note on numbers: the speedup here will be 2–4× on a typical laptop.
// On a system with more cores and a deeper cache hierarchy the gap is larger.
// If your numbers look different that's expected — cache topology varies
// significantly between machines.

static constexpr int  FS_THREADS    = 4;
static constexpr long FS_ITERATIONS = 100'000'000L;

// volatile forces a real memory write on every increment, preventing the
// compiler from accumulating the count in a register and writing back once.
// Without it the false sharing never happens — the cache line is barely
// touched — and the benchmark measures nothing useful.
struct UnalignedCounter { volatile long value = 0; };
struct alignas(64) AlignedCounter { volatile long value = 0; };

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

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  " << std::left  << std::setw(32) << label
              << std::right << std::setw(7) << std::fixed << std::setprecision(1)
              << ms << " ms\n";
    return ms;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SPSC queue throughput vs mutex-protected std::queue
// ─────────────────────────────────────────────────────────────────────────────
//
// This benchmark isolates queue overhead, not simulation throughput.
// The SPSC advantage only materialises at high submission rates — for
// coarse Monte Carlo batches (N=4 tasks per simulation run) the mutex
// version is fine and simpler. The crossover is roughly where queue
// operations become a measurable fraction of task execution time.

static constexpr int    QBench_ITEMS = 10'000'000;
static constexpr size_t QBench_CAP   = 1 << 14;  // 16384

double bench_spsc() {
    SPSCQueue<int, QBench_CAP> q;
    std::atomic<bool> done{false};
    long consumed = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::thread producer([&] {
        for (int i = 0; i < QBench_ITEMS; ++i) {
            while (!q.try_push(i)) {}  // spin on full — shouldn't happen at this rate
        }
        done.store(true, std::memory_order_release);
    });

    while (!done.load(std::memory_order_acquire) || q.size_approx() > 0) {
        int val;
        if (q.try_pop(val)) ++consumed;
    }
    // Drain anything left after producer signals done
    int val;
    while (q.try_pop(val)) ++consumed;

    producer.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mops = consumed / (ms / 1000.0) / 1e6;
    std::cout << "  " << std::left  << std::setw(32) << "SPSC lock-free queue"
              << std::right << std::setw(7) << std::fixed << std::setprecision(1)
              << ms << " ms  (" << std::setprecision(0) << mops << "M items/sec)\n";
    return ms;
}

double bench_mutex_queue() {
    // std::queue + mutex, same producer/consumer pattern
    std::queue<int> q;
    std::mutex      mtx;
    std::atomic<bool> done{false};
    long consumed = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::thread producer([&] {
        for (int i = 0; i < QBench_ITEMS; ++i) {
            std::lock_guard<std::mutex> lk(mtx);
            q.push(i);
        }
        done.store(true, std::memory_order_release);
    });

    while (true) {
        int val;
        bool got = false;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (!q.empty()) { val = q.front(); q.pop(); got = true; }
        }
        if (got) { ++consumed; continue; }
        if (done.load(std::memory_order_acquire)) break;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        while (!q.empty()) { q.pop(); ++consumed; }
    }

    producer.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mops = consumed / (ms / 1000.0) / 1e6;
    std::cout << "  " << std::left  << std::setw(32) << "Mutex-protected queue"
              << std::right << std::setw(7) << std::fixed << std::setprecision(1)
              << ms << " ms  (" << std::setprecision(0) << mops << "M items/sec)\n";
    return ms;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Model comparison and thread scaling
// ─────────────────────────────────────────────────────────────────────────────

template<typename Model>
void bench_model(const typename Model::Params& p, int paths, int threads) {
    auto result = run_simulation<Model>(p, paths, threads, 42);
    auto report = compute_risk(result.final_prices, p.S0);
    double tput = paths / (result.elapsed_ms / 1000.0) / 1e6;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::left  << std::setw(26) << Model::name()
              << " | " << std::right << std::setw(7) << result.elapsed_ms << " ms"
              << " | " << std::setw(4) << std::setprecision(1) << tput << "M p/s"
              << " | VaR95=$"  << std::setprecision(2) << report.var_95
              << "  CVaR95=$" << report.cvar_95 << "\n";
}

int main() {
    const int N = static_cast<int>(std::thread::hardware_concurrency());

    // ── 1. False sharing ──────────────────────────────────────────────────
    std::cout << "=== False Sharing (" << FS_THREADS << " threads, "
              << FS_ITERATIONS / 1'000'000 << "M increments/thread) ===\n";
    double t_unaligned = bench_false_sharing<UnalignedCounter>("Unaligned (false sharing)");
    double t_aligned   = bench_false_sharing<AlignedCounter>  ("alignas(64)  (no sharing)");
    std::cout << "  Speedup: " << std::setprecision(1) << (t_unaligned / t_aligned) << "x\n\n";

    // ── 2. SPSC vs mutex ──────────────────────────────────────────────────
    std::cout << "=== Queue Throughput (" << QBench_ITEMS / 1'000'000
              << "M items, 1 producer / 1 consumer) ===\n";
    double t_mutex = bench_mutex_queue();
    double t_spsc  = bench_spsc();
    std::cout << "  SPSC speedup: " << std::setprecision(1)
              << (t_mutex / t_spsc) << "x\n\n";

    // ── 3. GBM vs jump-diffusion ──────────────────────────────────────────
    const double S0 = 150.0, mu = 0.08, sigma = 0.20, T = 0.25;
    std::cout << "=== Model Comparison (1M paths, " << N << " threads) ===\n";
    GBM::Params        gbm_p{S0, mu, sigma, T, 252};
    MertonJump::Params jmp_p{S0, mu, sigma, T, 252, 3.0, -0.05, 0.08};
    bench_model<GBM>       (gbm_p, 1'000'000, N);
    bench_model<MertonJump>(jmp_p, 1'000'000, N);
    std::cout << "\n";

    // ── 4. Thread scaling ─────────────────────────────────────────────────
    std::cout << "=== Thread Scaling (GBM, 1M paths) ===\n";
    double t1_ms = 0;
    for (int t = 1; t <= N; ++t) {
        auto r = run_simulation<GBM>(gbm_p, 1'000'000, t, 42);
        if (t == 1) t1_ms = r.elapsed_ms;
        double tput = 1e6 / (r.elapsed_ms / 1000.0) / 1e6;
        std::cout << "  threads=" << std::setw(2) << t
                  << "  " << std::setw(7) << std::fixed << std::setprecision(1)
                  << r.elapsed_ms << " ms"
                  << "  (" << std::setprecision(1) << tput << "M p/s)"
                  << "  speedup=" << std::setprecision(2) << (t1_ms / r.elapsed_ms) << "x\n";
    }

    return 0;
}
