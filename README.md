# Monte Carlo Equity Simulator

A high-performance C++ simulator for equity price distributions. Runs millions of price paths under two models — Geometric Brownian Motion and Merton Jump-Diffusion — and outputs a risk report with VaR, CVaR, and a full percentile distribution.

The design priorities are the same as production low-latency systems: no heap allocation in the hot path, no shared mutable state between threads, and measurable performance at every layer.

## Models

**GBM** — the standard model, fast and analytically tractable. Assumes smooth continuous price moves; underestimates tail risk because it has no mechanism for sudden gaps.

**Merton Jump-Diffusion** — adds a Poisson-driven jump process on top of GBM. Captures earnings shocks, macro events, and any discontinuous move the diffusion term can't produce. On identical parameters, jump-diffusion consistently produces 35–40% higher VaR and CVaR — a difference that matters for position sizing.

```
$ ./build/simulator --price 150 --drift 0.08 --vol 0.20 --years 0.25

Model: GBM             VaR95=$20.82  CVaR95=$26.01
Model: Jump-Diffusion  VaR95=$28.63  CVaR95=$37.14  (+38%, +43%)
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/simulator --help
./build/bench       # runs all benchmarks
```

Requires: CMake ≥ 3.16, C++17, pthreads. No other dependencies.

## Performance (4-core machine)

**Thread scaling** — near-linear, 3.73× on 4 cores:

| Threads | Time | Throughput | Speedup |
|---|---|---|---|
| 1 | 4017 ms | 0.2M paths/sec | 1.0× |
| 2 | 2017 ms | 0.5M paths/sec | 2.0× |
| 4 | 1076 ms | 0.9M paths/sec | 3.7× |

**False sharing** — `alignas(64)` on per-thread state is 1.8× faster under contention. The benchmark uses `volatile` counters to force real memory writes; without it the compiler keeps everything in registers and the effect disappears entirely.

**SPSC queue** — 16× higher throughput than `std::mutex + std::queue` at 10M items/sec:

| | Throughput |
|---|---|
| Mutex queue | 13M items/sec |
| SPSC lock-free | 214M items/sec |

## Architecture

```
src/
  rng.hpp             xoshiro256++ — 3× faster than mt19937, 32-byte state
  task.hpp            SBO callable wrapper replacing std::function
  thread_pool.hpp     Hand-rolled pool: condition_variable + atomic pending counter
  simulator.hpp       Simulator<Model> template — zero virtual dispatch
  gbm.hpp             GBM model policy
  jump_diffusion.hpp  Merton jump-diffusion model policy
  stats.hpp           VaR, CVaR, percentile computation
  main.cpp            CLI
benchmark/
  bench.cpp           False sharing, SPSC vs mutex, model comparison, thread scaling
```

## Design decisions

**`task.hpp` — SBO callable, not `std::function`**

`std::function`'s SBO buffer size is implementation-defined and its heap fallback is not guaranteed to be absent. `Task` uses a fixed 64-byte inline buffer (one cache line). If the callable exceeds 64 bytes, a `static_assert` fires immediately with a message explaining what to do — not a cryptic linker error three files deep. A separate `move_` function pointer ensures move construction is correct for any callable type, not just trivially-copyable ones.

**`spsc_queue.hpp` — lock-free single-producer/consumer queue**

Head and tail indices are `uint64_t` and never wrap — at one billion operations per second, overflow takes 585 years. They sit on separate `alignas(64)` cache lines so the producer and consumer never invalidate each other's state. Capacity is a power-of-2 checked at compile time via `static_assert`; the queue uses bitmask indexing rather than modulo. `size_approx()` is deliberately named approximate — in a concurrent queue, an "exact" size is meaningless between the two atomic loads required to compute it.

**`thread_pool.hpp` — two condition variables**

`cv_` wakes workers when tasks arrive or the pool stops. `done_cv_` wakes the `wait()` caller when the atomic `pending_` counter hits zero. A single condition variable would force `wait()` to re-acquire the task mutex on every completion, adding contention on the critical path when tasks complete in rapid succession.

**`simulator.hpp` — `alignas(64)` on `WorkerState`**

Each thread's RNG state and work range live in one `WorkerState`. Without alignment, adjacent workers can share a cache line; every write by one core triggers a coherence invalidation visible to the other. The false sharing benchmark in `bench.cpp` makes this measurable: 1.8× slower without the alignment.

**`Simulator<Model>` template**

The model (GBM, MertonJump) is a compile-time template parameter, not a virtual base class. No vtable, no indirect call, no pointer indirection — the compiler sees the full implementation of `simulate_antithetic` at the call site and can inline and vectorise it. Adding a new model means writing a struct with two static methods; the simulator and benchmark pick it up with no other changes.

## The math

GBM exact discrete solution (no Euler-Maruyama error):
```
S(t+dt) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z),  Z ~ N(0,1)
```

The `σ²/2` Itô correction converts arithmetic drift to geometric drift so that `E[S(T)] = S₀·exp(μT)`.

Merton jump-diffusion adds a compound Poisson term:
```
S(t+dt) = S_GBM(t+dt) · exp(J·N),  N ~ Bernoulli(λ·dt),  J ~ N(μⱼ, σⱼ²)
```

The continuous drift is reduced by `λ·κ` (where `κ = E[eᴶ] - 1`) so total expected return stays equal to `μ` regardless of jump intensity.

Normal samples come from Box-Muller applied to xoshiro256++ uniform draws.
