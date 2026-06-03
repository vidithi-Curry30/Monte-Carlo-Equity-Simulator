#pragma once
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "csv_loader.hpp"
#include "gbm.hpp"
#include "jump_diffusion.hpp"
#include "simulator.hpp"
#include "stats.hpp"

// Side-by-side risk metrics from both models run on the same calibrated data.
struct ComparisonReport {
    RiskReport gbm;
    RiskReport merton;

    // The tail underestimation ratio: how much wider the Merton tail is vs GBM.
    // A ratio of 1.30 means GBM underestimates CVaR by 30%.
    // This is the direct answer to the central question.
    double cvar_ratio;  // merton.cvar_95 / gbm.cvar_95
    double var_ratio;   // merton.var_95  / gbm.var_95

    // Empirical moments from the historical data for comparison.
    EmpiricalMoments empirical;

    double elapsed_gbm_ms;
    double elapsed_merton_ms;
};

// Run both GBM and Merton on the same calibrated parameters and produce a
// side-by-side comparison. The same base seed is used for both so that
// differences in output are due to model structure, not RNG variance.
//
// This is the core diagnostic for the question: "Does assuming log-normal
// returns systematically underestimate downside risk, and by how much?"
// The cvar_ratio answers "by how much" directly.
inline ComparisonReport run_comparison(
    const std::vector<double>& historical_prices,
    double T,
    int    paths,
    int    steps,
    int    threads,
    uint64_t seed = 42)
{
    auto est     = estimate_params(historical_prices);
    auto empir   = empirical_moments(historical_prices);

    GBM::Params gbm_p{est.S0, est.mu, est.sigma, T, steps};
    MertonJump::Params mj_p{est.S0, est.mu, est.sigma, T, steps,
                             est.lambda, est.mu_j, est.sigma_j};

    auto gbm_res    = run_simulation<GBM>(gbm_p, paths, threads, seed);
    auto merton_res = run_simulation<MertonJump>(mj_p, paths, threads, seed);

    auto gbm_report    = compute_risk(gbm_res.final_prices,    est.S0);
    auto merton_report = compute_risk(merton_res.final_prices, est.S0);

    // Guard against divide-by-zero if GBM produces zero VaR (not realistic but defensive).
    const double cvar_ratio = (gbm_report.cvar_95 > 0)
        ? merton_report.cvar_95 / gbm_report.cvar_95 : 0.0;
    const double var_ratio  = (gbm_report.var_95  > 0)
        ? merton_report.var_95  / gbm_report.var_95  : 0.0;

    return ComparisonReport{
        gbm_report, merton_report,
        cvar_ratio, var_ratio,
        empir,
        gbm_res.elapsed_ms, merton_res.elapsed_ms
    };
}

static void print_comparison(const ComparisonReport& c, const std::string& source) {
    const auto& g = c.gbm;
    const auto& m = c.merton;
    std::cout << std::fixed;

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          GBM vs Merton Jump-Diffusion — Tail Comparison      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "  Source: " << source << "\n";
    std::cout << "  S0 = $" << std::setprecision(2) << g.S0 << "\n\n";

    // Column header
    std::cout << std::setw(28) << std::left  << "  Metric"
              << std::setw(14) << std::right << "GBM"
              << std::setw(14) << std::right << "Merton"
              << std::setw(12) << std::right << "Ratio"
              << "\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    auto row = [&](const char* label, double gv, double mv, bool is_ratio_interesting = true) {
        const double ratio = (gv != 0) ? mv / gv : 0.0;
        std::cout << "  " << std::setw(26) << std::left  << label
                  << std::setw(14) << std::right << std::setprecision(2) << gv
                  << std::setw(14) << std::right << mv;
        if (is_ratio_interesting)
            std::cout << std::setw(11) << std::right << std::setprecision(2) << ratio << "x";
        else
            std::cout << std::setw(12) << std::right << "—";
        std::cout << "\n";
    };

    row("Mean price ($)",          g.mean,    m.mean,    false);
    row("Std deviation ($)",       g.std_dev, m.std_dev, true);
    std::cout << "\n";
    row("P5  price ($)",           g.p5,      m.p5,      false);
    row("P1  price (= 99% VaR)",   g.S0 - g.var_99, m.S0 - m.var_99, false);
    std::cout << "\n";
    row("VaR  95% loss ($)",       g.var_95,  m.var_95,  true);
    row("VaR  99% loss ($)",       g.var_99,  m.var_99,  true);
    row("CVaR 95% loss ($)",       g.cvar_95, m.cvar_95, true);
    std::cout << "\n";

    // Distributional shape — this is where the models differ most visibly
    std::cout << "  " << std::string(66, '-') << "\n";
    std::cout << "  " << std::setw(26) << std::left << "Skewness"
              << std::setw(14) << std::right << std::setprecision(3) << g.skewness
              << std::setw(14) << std::right << m.skewness
              << std::setw(12) << std::right << "—" << "\n";
    std::cout << "  " << std::setw(26) << std::left << "Excess kurtosis"
              << std::setw(14) << std::right << g.excess_kurtosis
              << std::setw(14) << std::right << m.excess_kurtosis
              << std::setw(12) << std::right << "—" << "\n";
    std::cout << "  " << std::setw(26) << std::left << "Empirical (historical)"
              << std::setw(14) << std::right << "—"
              << std::setw(14) << std::right << "—"
              << std::setw(12) << std::right << "—" << "\n";
    std::cout << "  " << std::setw(26) << std::left << "  skewness"
              << std::setw(14) << std::right << std::setprecision(3) << c.empirical.skewness
              << "\n";
    std::cout << "  " << std::setw(26) << std::left << "  excess kurtosis"
              << std::setw(14) << std::right << c.empirical.excess_kurtosis
              << "\n";

    // The answer to the question
    std::cout << "\n  " << std::string(66, '=') << "\n";
    std::cout << "  Tail underestimation (Merton CVaR / GBM CVaR):  "
              << std::setprecision(2) << c.cvar_ratio << "x\n";

    if (c.cvar_ratio > 1.25)
        std::cout << "  >> GBM underestimates expected shortfall by "
                  << std::setprecision(0) << (c.cvar_ratio - 1.0) * 100.0
                  << "% — log-normal assumption is materially inadequate here.\n";
    else if (c.cvar_ratio > 1.10)
        std::cout << "  >> Moderate underestimation — log-normal is a reasonable\n"
                  << "     approximation but understates tail risk meaningfully.\n";
    else
        std::cout << "  >> Models agree closely — jump risk is low for this asset/horizon.\n";

    // Kurtosis gap — the mechanistic explanation for any CVaR gap
    const double kurt_gap = m.excess_kurtosis - g.excess_kurtosis;
    std::cout << "\n  Kurtosis gap (Merton - GBM):  "
              << std::setprecision(2) << kurt_gap
              << "  |  Empirical excess kurtosis:  "
              << c.empirical.excess_kurtosis << "\n";
    if (c.empirical.excess_kurtosis > m.excess_kurtosis + 0.5)
        std::cout << "  >> Merton still underfits the empirical tail — actual returns\n"
                  << "     are fatter-tailed than the calibrated jump-diffusion predicts.\n";
    else if (c.empirical.excess_kurtosis > g.excess_kurtosis + 0.5)
        std::cout << "  >> Merton captures most of the empirical fat-tail — GBM misses it.\n";

    std::cout << "  " << std::string(66, '=') << "\n\n";
}
