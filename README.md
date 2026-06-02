# Monte Carlo Equity Simulator

A high-performance C++ simulator for equity price distributions using Geometric Brownian Motion. Designed around the same performance priorities as production quant systems: minimal latency, no heap allocation in the hot path, and linear scaling across cores.

## What it does

Given a stock's current price, expected drift, and volatility, the simulator runs millions of possible future price paths and reports:

- Full percentile distribution of terminal prices (P5 → P95)
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional VaR / Expected Shortfall at 95%
- Probability of profit

## Example

```
$ ./build/simulator --price 150 --drift 0.08 --vol 0.20 --years 0.25

=== Monte Carlo Equity Simulator ===
  S0=150.00  mu=8.00%  sigma=20.00%  T=0.25yr
  paths=1000000  steps=252

Running simulation... done in 946 ms (1.1M paths/sec)

--- Price Distribution ---
  Mean final price : $153.03
  Median           : $152.27
  Std deviation    : $15.36

--- Percentiles ---
   5th pct  : $129.14  [############                  ]
  50th pct  : $152.27  [###############               ]
  95th pct  : $179.53  [#################             ]

--- Risk Metrics ---
  VaR  95%                      : $20.86
  VaR  99%                      : $29.39
  CVaR 95% (expected shortfall) : $26.07

--- Outcome ---
  Probability of profit : 56.0%  [################              ]
```

## Performance

Benchmarked on a 4-core machine, 2M paths × 252 steps:

| Mode | Time | Throughput |
|---|---|---|
| Single-threaded, standard MC | 10,384 ms | 0.2M paths/sec |
| Single-threaded, antithetic variates | 7,484 ms | 0.3M paths/sec |
| 4-thread OpenMP, antithetic variates | 1,908 ms | 1.0M paths/sec |

**5.4× end-to-end speedup** from single-threaded standard MC to parallel antithetic variates.

## Design decisions

**xoshiro256++ RNG** — Replaces `std::mt19937`. ~3× faster, passes PractRand and BigCrush, and is trivially seedable per-thread without locking. The period (2²⁵⁶−1) far exceeds any simulation budget.

**Antithetic variates** — For each random draw Z, a paired path uses −Z. The two paths are negatively correlated, which cancels first-order Monte Carlo variance. In practice: same accuracy as 2× the paths at ~1.4× the wall time.

**Thread-local RNG state** — Each OpenMP thread seeds its own `Xoshiro256pp` from a hash of the loop index. There is no shared mutable state on the hot path — no mutexes, no atomics, no false sharing on cache lines.

**Pre-allocated flat arrays** — `std::vector` is sized once before the parallel loop. No heap allocation occurs inside the simulation. Predictable memory access pattern enables hardware prefetching.

**Exact GBM discretisation** — Uses the log-normal solution `S(t+dt) = S(t) * exp((μ − σ²/2)dt + σ√dt · Z)` rather than an Euler–Maruyama approximation. Zero discretisation error regardless of step size.

**`-O3 -march=native -ffast-math`** — Enables auto-vectorisation and SIMD reductions. The inner `exp()` loop is the bottleneck; `ffast-math` allows the compiler to use SVML or its equivalent.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Requires: CMake ≥ 3.16, a C++17 compiler. OpenMP is optional but recommended.

```bash
./build/simulator --help
./build/bench
```

## The math

Price follows Geometric Brownian Motion:

```
dS = S · (μ dt + σ dW)
```

where `dW = sqrt(dt) · Z`, `Z ~ N(0,1)`. The exact discrete solution is:

```
S(t+dt) = S(t) · exp((μ - σ²/2) · dt + σ · sqrt(dt) · Z)
```

The `σ²/2` term is the Itô correction — it converts the arithmetic drift to the geometric drift needed for the log-normal distribution. This is the same model underlying the Black-Scholes options pricing formula.

Normal samples are generated via the Box-Muller transform from uniform draws.
