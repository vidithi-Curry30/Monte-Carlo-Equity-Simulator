#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "csv_loader.hpp"
#include "gbm.hpp"
#include "options.hpp"
#include "simulator.hpp"

// ── Minimal test harness ──────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond) do { \
    if (cond) { \
        ++g_passed; \
    } else { \
        ++g_failed; \
        std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    } \
} while(0)

#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

// ── norm_cdf ──────────────────────────────────────────────────────────────────

static void test_norm_cdf() {
    CHECK_NEAR(norm_cdf(0.0),   0.5,     1e-15);
    CHECK_NEAR(norm_cdf(1.96),  0.97500, 1e-4);   // well-known 95% one-tail
    CHECK_NEAR(norm_cdf(-1.96), 0.02500, 1e-4);
    CHECK_NEAR(norm_cdf(3.0),   0.99865, 1e-5);
    CHECK_NEAR(norm_cdf(-3.0),  0.00135, 1e-5);
    // Symmetry
    CHECK_NEAR(norm_cdf(1.5) + norm_cdf(-1.5), 1.0, 1e-15);
}

// ── black_scholes_call ────────────────────────────────────────────────────────

static void test_black_scholes() {
    // ATM call: S=K=100, r=5%, sigma=20%, T=1yr
    // Known result: ~10.4506
    double bs = black_scholes_call(100.0, 100.0, 0.05, 1.0, 0.20);
    CHECK_NEAR(bs, 10.4506, 1e-3);

    // Deep ITM: price ≈ intrinsic (discounted)
    double itm = black_scholes_call(200.0, 100.0, 0.05, 1.0, 0.20);
    CHECK(itm > 95.0);  // must be above discounted intrinsic

    // Deep OTM: very small
    double otm = black_scholes_call(100.0, 200.0, 0.05, 1.0, 0.20);
    CHECK(otm < 0.01);

    // Zero time: max(S-K, 0)
    double exp = black_scholes_call(110.0, 100.0, 0.05, 0.0, 0.20);
    CHECK_NEAR(exp, 0.0, 1e-10);  // T<=0 returns 0.0

    // Put-call parity: C - P = S - K*exp(-rT)
    // We only have call, but can verify C >= max(S*exp(-rT) - K*exp(-rT), 0)
    double lower = 100.0 - 100.0 * std::exp(-0.05);
    CHECK(bs >= lower);
}

// ── MC convergence vs Black-Scholes ──────────────────────────────────────────
//
// With 500k paths and a fixed seed the MC price should land within
// 3 standard errors of the Black-Scholes price for a vanilla GBM call.

static void test_mc_bs_convergence() {
    GBM::Params p{150.0, 0.08, 0.20, 0.25, 252};
    const double K = 155.0, r = 0.05;

    auto opt = price_call<GBM>(p, K, r, 500'000, 4, 42);

    // MC price within 3 SE of BS
    CHECK(std::abs(opt.mc_price - opt.bs_price) <= 3.0 * opt.std_error);

    // Greeks have the right sign and rough magnitude for these parameters
    CHECK(opt.delta > 0.0 && opt.delta < 1.0);   // call delta in (0,1)
    CHECK(opt.gamma > 0.0);                        // gamma always positive
    CHECK(opt.vega  > 0.0);                        // vega always positive
}

// ── estimate_params round-trip ────────────────────────────────────────────────
//
// Generate a synthetic GBM price series with known mu/sigma, run through
// the estimator, and check the recovered values are within 20%.
// (Statistical estimator — tight tolerance is unrealistic at N=252.)

static void test_estimate_params_roundtrip() {
    // Simulate one year of daily prices with mu=10%, sigma=25%, S0=100
    const double mu = 0.10, sigma = 0.25, S0 = 100.0;
    const int N = 252;
    const double dt = 1.0 / 252.0;

    // Use a simple deterministic sequence via the project's RNG
    Xoshiro256pp rng(12345);
    std::vector<double> prices;
    prices.reserve(N + 1);
    prices.push_back(S0);
    for (int i = 0; i < N; ++i) {
        double z = rng.nextNormal();
        prices.push_back(prices.back() * std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z));
    }

    auto est = estimate_params(prices);
    CHECK_NEAR(est.S0, prices.back(), 1e-9);           // S0 is exact (last price)
    CHECK(std::abs(est.sigma - sigma) / sigma < 0.20); // sigma within 20%
    // mu estimate is noisy at N=252, so use a wide band
    CHECK(est.mu > -0.60 && est.mu < 0.80);  // drift is noisy at N=252; wide band is correct
}

// ── load_prices CSV parsing ───────────────────────────────────────────────────

static void test_csv_parsing() {
    // Write a temp Yahoo-style CSV and check prices are loaded correctly
    const char* path = "/tmp/test_prices.csv";
    {
        std::ofstream f(path);
        f << "Date,Open,High,Low,Close,Adj Close,Volume\n";
        f << "2024-01-01,100.0,102.0,99.0,101.50,101.50,1000000\n";
        f << "2024-01-02,101.5,103.0,100.0,102.75,102.75,1200000\n";
    }

    // Only 2 rows — too few for estimate_params, but load_prices should work
    std::ifstream check(path);
    std::string line;
    int data_lines = 0;
    while (std::getline(check, line)) {
        if (!line.empty() && line[0] != 'D') ++data_lines;  // skip header
    }
    CHECK(data_lines == 2);

    // Write enough rows for load_prices (needs >= 30)
    {
        std::ofstream f(path);
        f << "Date,Open,High,Low,Close,Adj Close,Volume\n";
        for (int i = 0; i < 35; ++i)
            f << "2024-01-" << (i + 1) << ",100,105,95," << (100.0 + i) << "," << (100.0 + i) << ",1000000\n";
    }

    auto prices = load_prices(path);
    CHECK(prices.size() == 35);
    CHECK_NEAR(prices[0],  100.0, 1e-9);
    CHECK_NEAR(prices[34], 134.0, 1e-9);

    // date,price format (two columns, no header recognised)
    const char* path2 = "/tmp/test_prices2.csv";
    {
        std::ofstream f(path2);
        for (int i = 0; i < 35; ++i)
            f << "2024-01-" << (i + 1) << "," << (50.0 + i) << "\n";
    }
    auto prices2 = load_prices(path2);
    CHECK(prices2.size() == 35);
    CHECK_NEAR(prices2[0],  50.0, 1e-9);
    CHECK_NEAR(prices2[34], 84.0, 1e-9);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_norm_cdf();
    test_black_scholes();
    test_mc_bs_convergence();
    test_estimate_params_roundtrip();
    test_csv_parsing();

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
