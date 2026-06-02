#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include "gbm.hpp"
#include "jump_diffusion.hpp"
#include "simulator.hpp"
#include "stats.hpp"

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " --price <S0> --drift <mu> --vol <sigma> --years <T> [options]\n"
        << "\nRequired:\n"
        << "  --price   Initial stock price       (e.g. 150.0)\n"
        << "  --drift   Annualised drift, decimal  (e.g. 0.08)\n"
        << "  --vol     Annualised volatility       (e.g. 0.20)\n"
        << "  --years   Horizon in years            (e.g. 0.25)\n"
        << "\nOptional:\n"
        << "  --model   gbm | jump                (default: gbm)\n"
        << "  --paths   Simulation paths           (default: 1000000)\n"
        << "  --steps   Time steps per path        (default: 252)\n"
        << "  --threads Worker threads             (default: hardware concurrency)\n"
        << "  --seed    RNG seed                   (default: 42)\n"
        << "\nJump-diffusion parameters (--model jump):\n"
        << "  --lambda  Jump intensity, jumps/year (default: 3.0)\n"
        << "  --muj     Mean log-jump size         (default: -0.05)\n"
        << "  --sigmaj  Std dev of log-jump size   (default: 0.08)\n"
        << "\nExamples:\n"
        << "  " << prog << " --price 150 --drift 0.08 --vol 0.20 --years 0.25\n"
        << "  " << prog << " --price 150 --drift 0.08 --vol 0.20 --years 0.25 "
                           "--model jump --lambda 4 --muj -0.06\n";
}

static void print_bar(double fraction, int width = 28) {
    int filled = static_cast<int>(fraction * width);
    std::cout << "[";
    for (int i = 0; i < width; ++i) std::cout << (i < filled ? '#' : ' ');
    std::cout << "]";
}

static void print_report(const RiskReport& r, const char* model_name,
                         double elapsed_ms, int paths, int threads) {
    const double throughput = paths / (elapsed_ms / 1000.0) / 1e6;
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\nModel  : " << model_name << "\n";
    std::cout << "Threads: " << threads
              << "  |  " << paths / 1'000'000.0 << "M paths"
              << "  |  " << elapsed_ms << " ms"
              << "  (" << std::setprecision(1) << throughput << "M paths/sec)\n";

    std::cout << std::setprecision(2);
    std::cout << "\n--- Price Distribution (S0 = $" << r.S0 << ") ---\n";
    std::cout << "  Mean            $" << r.mean   << "\n";
    std::cout << "  Median          $" << r.median << "\n";
    std::cout << "  Std deviation   $" << r.std_dev << "\n";

    std::cout << "\n--- Percentiles ---\n";
    auto row = [&](const char* label, double price) {
        std::cout << "  " << std::setw(8) << std::left << label
                  << " $" << std::setw(8) << std::right << price << "  ";
        print_bar(price / (r.S0 * 2.0));
        std::cout << "\n";
    };
    row("P5",  r.p5);
    row("P10", r.p10);
    row("P25", r.p25);
    row("P50", r.median);
    row("P75", r.p75);
    row("P90", r.p90);
    row("P95", r.p95);

    std::cout << "\n--- Risk Metrics ---\n";
    std::cout << "  VaR  95%  (max loss, 95% conf)  $" << r.var_95  << "\n";
    std::cout << "  VaR  99%                         $" << r.var_99  << "\n";
    std::cout << "  CVaR 95%  (expected shortfall)   $" << r.cvar_95 << "\n";

    std::cout << "\n--- Outcome ---\n";
    std::cout << "  P(profit)  " << std::setprecision(1)
              << r.prob_profit * 100.0 << "%  ";
    print_bar(r.prob_profit);
    std::cout << "\n\n";
}

int main(int argc, char* argv[]) {
    double   S0      = 0, mu = 0, sigma = 0, T = 0;
    int      paths   = 1'000'000;
    int      steps   = 252;
    int      threads = static_cast<int>(std::thread::hardware_concurrency());
    uint64_t seed    = 42;
    std::string model = "gbm";
    // Jump-diffusion extras
    double lambda = 3.0, mu_j = -0.05, sigma_j = 0.08;

    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "--help") { print_usage(argv[0]); return 0; }
        if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
        if      (f == "--price")   S0      = std::atof(argv[++i]);
        else if (f == "--drift")   mu      = std::atof(argv[++i]);
        else if (f == "--vol")     sigma   = std::atof(argv[++i]);
        else if (f == "--years")   T       = std::atof(argv[++i]);
        else if (f == "--paths")   paths   = std::atoi(argv[++i]);
        else if (f == "--steps")   steps   = std::atoi(argv[++i]);
        else if (f == "--threads") threads = std::atoi(argv[++i]);
        else if (f == "--seed")    seed    = std::stoull(argv[++i]);
        else if (f == "--model")   model   = argv[++i];
        else if (f == "--lambda")  lambda  = std::atof(argv[++i]);
        else if (f == "--muj")     mu_j    = std::atof(argv[++i]);
        else if (f == "--sigmaj")  sigma_j = std::atof(argv[++i]);
        else { std::cerr << "Unknown flag: " << f << "\n"; print_usage(argv[0]); return 1; }
    }

    if (S0 <= 0 || sigma <= 0 || T <= 0) { print_usage(argv[0]); return 1; }
    if (threads < 1) threads = 1;

    std::cout << "=== Monte Carlo Equity Simulator ===\n";
    std::cout << "S0=$" << S0 << "  mu=" << mu*100 << "%"
              << "  sigma=" << sigma*100 << "%  T=" << T << "yr\n";

    if (model == "jump") {
        std::cout << "lambda=" << lambda << "  mu_j=" << mu_j
                  << "  sigma_j=" << sigma_j << "\n";
        MertonJump::Params p{S0, mu, sigma, T, steps, lambda, mu_j, sigma_j};
        auto result = run_simulation<MertonJump>(p, paths, threads, seed);
        auto report = compute_risk(result.final_prices, S0);
        print_report(report, MertonJump::name(), result.elapsed_ms, paths, threads);
    } else {
        GBM::Params p{S0, mu, sigma, T, steps};
        auto result = run_simulation<GBM>(p, paths, threads, seed);
        auto report = compute_risk(result.final_prices, S0);
        print_report(report, GBM::name(), result.elapsed_ms, paths, threads);
    }

    return 0;
}
