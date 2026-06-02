#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include "csv_loader.hpp"
#include "gbm.hpp"
#include "jump_diffusion.hpp"
#include "options.hpp"
#include "simulator.hpp"
#include "stats.hpp"

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [--csv file | --price S0 --drift mu --vol sigma] --years T [options]\n"
        << "\nPrice source (one required):\n"
        << "  --csv     CSV file of historical prices (estimates mu, sigma, S0)\n"
        << "  --price   Initial price            (e.g. 150.0)\n"
        << "  --drift   Annualised drift          (e.g. 0.08)\n"
        << "  --vol     Annualised volatility      (e.g. 0.20)\n"
        << "\nRequired:\n"
        << "  --years   Horizon in years           (e.g. 0.25)\n"
        << "\nSimulation options:\n"
        << "  --model   gbm | jump               (default: gbm)\n"
        << "  --paths   Simulation paths          (default: 1000000)\n"
        << "  --steps   Time steps per path       (default: 252)\n"
        << "  --threads Worker threads            (default: hardware concurrency)\n"
        << "  --seed    RNG seed                  (default: 42)\n"
        << "\nOption pricing (optional):\n"
        << "  --option  call                      (enables option pricing)\n"
        << "  --strike  Option strike price       (e.g. 155.0)\n"
        << "  --rate    Risk-free rate, decimal   (e.g. 0.05)\n"
        << "\nJump-diffusion parameters (--model jump, overrides CSV estimates):\n"
        << "  --lambda  Jump intensity, jumps/yr  (default: 3.0, or from CSV)\n"
        << "  --muj     Mean log-jump size        (default: -0.05, or from CSV)\n"
        << "  --sigmaj  Std dev of log-jump size  (default: 0.08, or from CSV)\n"
        << "\nExamples:\n"
        << "  " << prog << " --price 150 --drift 0.08 --vol 0.20 --years 0.25\n"
        << "  " << prog << " --csv aapl.csv --years 0.25 --model jump\n"
        << "  " << prog << " --csv aapl.csv --years 0.25 --option call --strike 155 --rate 0.05\n";
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
    row("P5",  r.p5);   row("P10", r.p10);
    row("P25", r.p25);  row("P50", r.median);
    row("P75", r.p75);  row("P90", r.p90);
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

static void print_option(const OptionResult& opt, double K, double r, bool is_gbm) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- Option Pricing (European Call, K=$"
              << std::setprecision(2) << K
              << ", r=" << r * 100 << "%) ---\n";
    std::cout << std::setprecision(4);
    std::cout << "  MC price       $" << opt.mc_price
              << "  (SE ±$" << opt.std_error << ")\n";
    if (is_gbm && opt.bs_price > 0) {
        std::cout << "  Black-Scholes  $" << opt.bs_price
                  << "  (diff: " << std::showpos << opt.bs_error << std::noshowpos << ")\n";
        const double rel = std::abs(opt.bs_error) / opt.bs_price * 100.0;
        if (rel < 1.0)
            std::cout << "  ✓ Within 1% of Black-Scholes — simulation validated\n";
        else
            std::cout << "  Δ vs Black-Scholes: " << std::setprecision(1) << rel << "% "
                      << "(run with more --paths to reduce MC error)\n";
    }
    std::cout << "\n--- Greeks (finite differences) ---\n";
    std::cout << std::setprecision(4);
    std::cout << "  Delta  " << opt.delta << "  (price change per $1 move in S0)\n";
    std::cout << "  Gamma  " << opt.gamma << "  (delta change per $1 move in S0)\n";
    std::cout << "  Vega   $" << opt.vega  << "  (price change per 1% move in vol)\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    // Price source
    double      S0    = 0, mu = 0, sigma = 0;
    std::string csv_path;
    // Simulation
    double      T       = 0;
    int         paths   = 1'000'000;
    int         steps   = 252;
    int         threads = static_cast<int>(std::thread::hardware_concurrency());
    uint64_t    seed    = 42;
    std::string model   = "gbm";
    // Jump params (can be overridden by CSV or explicit flags)
    double lambda = 0, mu_j = 0, sigma_j = 0;
    bool   jump_params_set = false;
    // Option pricing
    std::string option_type;
    double      strike = 0, rate = 0;

    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "--help") { print_usage(argv[0]); return 0; }
        if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
        if      (f == "--price")   S0      = std::atof(argv[++i]);
        else if (f == "--drift")   mu      = std::atof(argv[++i]);
        else if (f == "--vol")     sigma   = std::atof(argv[++i]);
        else if (f == "--csv")     csv_path = argv[++i];
        else if (f == "--years")   T       = std::atof(argv[++i]);
        else if (f == "--paths")   paths   = std::atoi(argv[++i]);
        else if (f == "--steps")   steps   = std::atoi(argv[++i]);
        else if (f == "--threads") threads = std::atoi(argv[++i]);
        else if (f == "--seed")    seed    = std::stoull(argv[++i]);
        else if (f == "--model")   model   = argv[++i];
        else if (f == "--lambda")  { lambda  = std::atof(argv[++i]); jump_params_set = true; }
        else if (f == "--muj")     { mu_j    = std::atof(argv[++i]); jump_params_set = true; }
        else if (f == "--sigmaj")  { sigma_j = std::atof(argv[++i]); jump_params_set = true; }
        else if (f == "--option")  option_type = argv[++i];
        else if (f == "--strike")  strike  = std::atof(argv[++i]);
        else if (f == "--rate")    rate    = std::atof(argv[++i]);
        else { std::cerr << "Unknown flag: " << f << "\n"; print_usage(argv[0]); return 1; }
    }

    if (threads < 1) threads = 1;

    // ── Resolve parameters ────────────────────────────────────────────────────
    if (!csv_path.empty()) {
        try {
            auto prices   = load_prices(csv_path);
            auto est      = estimate_params(prices);
            S0    = est.S0;
            mu    = est.mu;
            sigma = est.sigma;
            // Only use CSV jump params if the user didn't supply them explicitly
            if (!jump_params_set) {
                lambda  = est.lambda;
                mu_j    = est.mu_j;
                sigma_j = est.sigma_j;
            }
            std::cout << "Loaded " << est.n_obs << " prices from " << csv_path << "\n";
            std::cout << "Estimated  mu=" << std::fixed << std::setprecision(1)
                      << mu * 100 << "%"
                      << "  sigma=" << sigma * 100 << "%"
                      << "  S0=$"   << std::setprecision(2) << S0 << "\n";
            if (model == "jump")
                std::cout << "Jump est.  lambda=" << std::setprecision(1) << lambda
                          << "  mu_j=" << std::setprecision(3) << mu_j
                          << "  sigma_j=" << sigma_j
                          << "  (" << est.n_jumps << " jumps detected)\n";
        } catch (const std::exception& e) {
            std::cerr << "Error loading CSV: " << e.what() << "\n";
            return 1;
        }
    } else {
        // Default jump params if not set by flags
        if (!jump_params_set) { lambda = 3.0; mu_j = -0.05; sigma_j = 0.08; }
    }

    if (S0 <= 0 || sigma <= 0 || T <= 0) { print_usage(argv[0]); return 1; }

    const bool do_option = (option_type == "call" && strike > 0);

    std::cout << "\n=== Monte Carlo Equity Simulator ===\n";
    std::cout << "S0=$"   << std::fixed << std::setprecision(2) << S0
              << "  mu="  << mu * 100 << "%"
              << "  sigma=" << sigma * 100 << "%"
              << "  T="   << T << "yr\n";

    // ── Run simulation ────────────────────────────────────────────────────────
    if (model == "jump") {
        std::cout << "lambda=" << lambda << "  mu_j=" << mu_j
                  << "  sigma_j=" << sigma_j << "\n";
        MertonJump::Params p{S0, mu, sigma, T, steps, lambda, mu_j, sigma_j};

        if (do_option) {
            // price_call reruns the simulation internally for Greeks bumps;
            // print risk report from its base run.
            auto opt = price_call<MertonJump>(p, strike, rate, paths, threads, seed);
            // Re-run once for the risk report (same seed = same prices)
            auto result = run_simulation<MertonJump>(p, paths, threads, seed);
            auto report = compute_risk(result.final_prices, S0);
            print_report(report, MertonJump::name(), result.elapsed_ms, paths, threads);
            print_option(opt, strike, rate, false);
        } else {
            auto result = run_simulation<MertonJump>(p, paths, threads, seed);
            auto report = compute_risk(result.final_prices, S0);
            print_report(report, MertonJump::name(), result.elapsed_ms, paths, threads);
        }
    } else {
        GBM::Params p{S0, mu, sigma, T, steps};

        if (do_option) {
            auto opt = price_call<GBM>(p, strike, rate, paths, threads, seed);
            auto result = run_simulation<GBM>(p, paths, threads, seed);
            auto report = compute_risk(result.final_prices, S0);
            print_report(report, GBM::name(), result.elapsed_ms, paths, threads);
            print_option(opt, strike, rate, true);
        } else {
            auto result = run_simulation<GBM>(p, paths, threads, seed);
            auto report = compute_risk(result.final_prices, S0);
            print_report(report, GBM::name(), result.elapsed_ms, paths, threads);
        }
    }

    return 0;
}
