# Monte Carlo Equity Simulator

High-performance C++ simulator for equity price distributions. Runs millions of paths under GBM and Merton Jump-Diffusion, outputs VaR/CVaR and option Greeks.

## Quick Start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Risk report
./build/simulator --price 150 --drift 0.08 --vol 0.20 --years 0.25

# From CSV (Yahoo Finance format)
./build/simulator --csv aapl.csv --years 0.25 --model jump

# European call pricing + Greeks
./build/simulator --price 150 --drift 0.08 --vol 0.20 --years 0.25 \
                  --option call --strike 155 --rate 0.05
```

Requires: CMake ≥ 3.16, C++17, pthreads. No other dependencies.

## Features

- **Two models** — GBM and Merton Jump-Diffusion. Jump-diffusion produces 35–40% higher VaR/CVaR on identical parameters, capturing earnings shocks that diffusion alone misses.
- **Option pricing** — Monte Carlo European call price with Black-Scholes validation and finite-difference Greeks (delta, gamma, vega) via common random numbers.
- **CSV calibration** — estimates μ, σ, and jump parameters from historical prices; supports Yahoo Finance, date+price, and single-column formats.
- **Performance** — near-linear thread scaling (3.7× on 4 cores), xoshiro256++ RNG, antithetic variates, zero heap allocation in the hot path.

## Architecture

| File | Purpose |
|---|---|
| `rng.hpp` | xoshiro256++ — 3× faster than mt19937, Box-Muller normals |
| `task.hpp` | SBO callable (64-byte inline buffer) replacing `std::function` |
| `thread_pool.hpp` | Hand-rolled pool: `condition_variable` + atomic pending counter |
| `spsc_queue.hpp` | Lock-free SPSC queue — benchmarked at 16× mutex throughput; suited for single feed pipelines, not the thread pool (which uses a locked deque for multi-consumer access) |
| `simulator.hpp` | `Simulator<Model>` template — zero virtual dispatch, inlines model |
| `csv_loader.hpp` | Price loading + parameter estimation from historical data |
| `options.hpp` | Black-Scholes, MC pricer, finite-difference Greeks |
| `benchmark/bench.cpp` | False sharing, SPSC vs mutex, thread scaling |
