#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "simulator.hpp"
#include "stats.hpp"

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " --price <S0> --drift <mu> --vol <sigma> --years <T>"
           " [--paths N] [--steps M] [--seed S]\n"
        << "\n"
        << "  --price   Initial stock price (e.g. 150.0)\n"
        << "  --drift   Annualised drift, decimal (e.g. 0.08 for 8%)\n"
        << "  --vol     Annualised volatility, decimal (e.g. 0.20 for 20%)\n"
        << "  --years   Time horizon in years (e.g. 0.25 for 3 months)\n"
        << "  --paths   Number of simulation paths (default: 1000000)\n"
        << "  --steps   Time steps per path (default: 252)\n"
        << "  --seed    RNG seed (default: 42)\n"
        << "\nExample:\n"
        << "  " << prog << " --price 150 --drift 0.08 --vol 0.20 --years 0.25\n";
}

static void print_bar(double fraction, int width = 30) {
    int filled = static_cast<int>(fraction * width);
    std::cout << "[";
    for (int i = 0; i < width; ++i) std::cout << (i < filled ? '#' : ' ');
    std::cout << "]";
}

int main(int argc, char* argv[]) {
    // Defaults
    double S0       = 0.0;
    double mu       = 0.0;
    double sigma    = 0.0;
    double T        = 0.0;
    int    paths    = 1'000'000;
    int    steps    = 252;
    uint64_t seed   = 42;
    bool   got_all  = false;

    // Simple flag parser — no external dependencies.
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc && flag != "--help") { print_usage(argv[0]); return 1; }
        if      (flag == "--price")  { S0    = std::atof(argv[++i]); }
        else if (flag == "--drift")  { mu    = std::atof(argv[++i]); }
        else if (flag == "--vol")    { sigma = std::atof(argv[++i]); }
        else if (flag == "--years")  { T     = std::atof(argv[++i]); }
        else if (flag == "--paths")  { paths = std::atoi(argv[++i]); }
        else if (flag == "--steps")  { steps = std::atoi(argv[++i]); }
        else if (flag == "--seed")   { seed  = std::stoull(argv[++i]); }
        else if (flag == "--help")   { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown flag: " << flag << "\n"; print_usage(argv[0]); return 1; }
    }

    got_all = (S0 > 0 && sigma > 0 && T > 0);
    if (!got_all) { print_usage(argv[0]); return 1; }

    const GBMParams params{S0, mu, sigma, T, steps};

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Monte Carlo Equity Simulator ===\n";
    std::cout << "  S0=" << S0 << "  mu=" << (mu*100) << "%  sigma="
              << (sigma*100) << "%  T=" << T << "yr\n";
    std::cout << "  paths=" << paths << "  steps=" << steps << "\n\n";
    std::cout << "Running simulation..." << std::flush;

    auto result = run_simulation(params, paths, seed);
    auto report = compute_risk(result.final_prices, S0);

    const double throughput = paths / (result.elapsed_ms / 1000.0) / 1e6;
    std::cout << " done in " << result.elapsed_ms << " ms"
              << " (" << std::setprecision(1) << throughput << "M paths/sec)\n\n";

    std::cout << std::setprecision(2);
    std::cout << "--- Price Distribution ---\n";
    std::cout << "  Mean final price : $" << report.mean << "\n";
    std::cout << "  Median           : $" << report.median << "\n";
    std::cout << "  Std deviation    : $" << report.std_dev << "\n\n";

    std::cout << "--- Percentiles ---\n";
    std::cout << "   5th pct  : $" << report.p5  << "  ";  print_bar(report.p5  / (S0*2)); std::cout << "\n";
    std::cout << "  10th pct  : $" << report.p10 << "  ";  print_bar(report.p10 / (S0*2)); std::cout << "\n";
    std::cout << "  25th pct  : $" << report.p25 << "  ";  print_bar(report.p25 / (S0*2)); std::cout << "\n";
    std::cout << "  50th pct  : $" << report.median << "  "; print_bar(report.median / (S0*2)); std::cout << "\n";
    std::cout << "  75th pct  : $" << report.p75 << "  ";  print_bar(report.p75 / (S0*2)); std::cout << "\n";
    std::cout << "  90th pct  : $" << report.p90 << "  ";  print_bar(report.p90 / (S0*2)); std::cout << "\n";
    std::cout << "  95th pct  : $" << report.p95 << "  ";  print_bar(report.p95 / (S0*2)); std::cout << "\n\n";

    std::cout << "--- Risk Metrics ---\n";
    std::cout << "  VaR  95% (max loss, 95% conf) : $" << report.var_95 << "\n";
    std::cout << "  VaR  99%                      : $" << report.var_99 << "\n";
    std::cout << "  CVaR 95% (expected shortfall) : $" << report.cvar_95 << "\n\n";

    std::cout << "--- Outcome ---\n";
    const double pct_profit = report.prob_profit * 100.0;
    std::cout << "  Probability of profit : " << std::setprecision(1)
              << pct_profit << "%  ";
    print_bar(report.prob_profit);
    std::cout << "\n\n";

    return 0;
}
