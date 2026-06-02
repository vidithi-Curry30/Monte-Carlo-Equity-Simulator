#pragma once
#include <chrono>
#include <cstdint>
#include <vector>
#include "rng.hpp"
#include "thread_pool.hpp"

struct SimResult {
    std::vector<double> final_prices;
    double elapsed_ms;
};

// Per-thread worker state.
//
// alignas(64) is the critical detail: a typical x86 cache line is 64 bytes.
// Without alignment, adjacent WorkerState objects may share a cache line.
// When thread A writes its rng state and thread B writes its results slice,
// both cores send "I own this cache line" invalidation messages to each other
// on every write — even though they're accessing different variables.
// This is false sharing: the penalty is ~100x slower memory access on the
// contended line. Alignment guarantees each WorkerState occupies its own
// cache line(s) with no cross-thread overlap.
struct alignas(64) WorkerState {
    Xoshiro256pp rng;
    int          begin;   // index of first path assigned to this worker
    int          end;     // one past the last path
};

template<typename Model>
SimResult run_simulation(
    const typename Model::Params& params,
    int      num_paths,
    int      n_threads,
    uint64_t base_seed = 42)
{
    std::vector<double> prices(num_paths);

    // Divide paths as evenly as possible across threads.
    // Integer division truncates: give the remainder to the last thread.
    const int batch = num_paths / n_threads;
    std::vector<WorkerState> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int begin = t * batch;
        const int end   = (t == n_threads - 1) ? num_paths : begin + batch;
        // Seed each thread with a unique hash of (base_seed, thread_id).
        // Multiplying by a large odd constant (Knuth's multiplicative hash)
        // ensures seeds that differ by 1 produce uncorrelated RNG streams.
        const uint64_t seed = base_seed + static_cast<uint64_t>(t) * 2654435761ULL;
        workers.push_back(WorkerState{Xoshiro256pp(seed), begin, end});
    }

    ThreadPool pool(n_threads);
    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < n_threads; ++t) {
        // Capture by pointer: WorkerState is non-copyable (Xoshiro256pp has
        // no copy constructor), and the vector outlives all tasks.
        WorkerState* ws = &workers[t];
        double*      out = prices.data();

        pool.submit([ws, out, &params] {
            int i = ws->begin;
            // Process pairs of paths with antithetic variates.
            for (; i + 1 < ws->end; i += 2) {
                auto [p_pos, p_neg] = Model::simulate_antithetic(params, ws->rng);
                out[i]     = p_pos;
                out[i + 1] = p_neg;
            }
            // Handle a trailing odd path if this thread's range is odd-sized.
            if (i < ws->end)
                out[i] = Model::simulate_path(params, ws->rng);
        });
    }

    pool.wait();

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    return SimResult{std::move(prices), elapsed_ms};
}
