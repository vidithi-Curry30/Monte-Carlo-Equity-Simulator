#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Estimated model parameters derived from historical price data.
struct ParamEstimates {
    double S0;       // last observed price (used as simulation starting point)
    double mu;       // annualised drift from log returns
    double sigma;    // annualised volatility from log returns
    double lambda;   // jump intensity: identified jumps per year
    double mu_j;     // mean log-jump size
    double sigma_j;  // std dev of log-jump size
    int    n_obs;    // number of price observations loaded
    int    n_jumps;  // number of returns classified as jumps
};

// Scan the header line for a column named "close", "adj close", or "price".
// Returns the 0-based index, or -1 if not found.
static int find_close_column(const std::string& header) {
    std::stringstream ss(header);
    std::string field;
    int idx = 0;
    while (std::getline(ss, field, ',')) {
        std::transform(field.begin(), field.end(), field.begin(), ::tolower);
        field.erase(0, field.find_first_not_of(" \t\r"));
        auto end = field.find_last_not_of(" \t\r");
        if (end != std::string::npos) field = field.substr(0, end + 1);
        if (field == "close" || field == "adj close" || field == "price") return idx;
        ++idx;
    }
    return -1;
}

// Split a CSV line into fields.
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, ',')) fields.push_back(f);
    return fields;
}

// Load closing prices from a CSV file.
//
// Supported formats:
//   - Single column of prices (no header)
//   - date,price  (two columns, no header)
//   - Yahoo Finance export: Date,Open,High,Low,Close,Adj Close,Volume
//     (header detected automatically; "Close" column is used)
//
// Lines where the first field cannot be parsed as a number are treated as
// headers. If a header contains a "Close" or "Price" column name, that
// column is used for all subsequent rows. Otherwise the last parseable
// numeric field on each line is used — which works for date,price but
// not for Yahoo format (where the last column is Volume).
std::vector<double> load_prices(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open file: " + path);

    std::vector<double> prices;
    std::string line;
    int close_col = -1;  // set once we find a header or infer from data

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto fields = split_csv(line);
        if (fields.empty()) continue;

        // Try to parse the first field as a number.
        // If it fails, this is a header row.
        bool first_is_numeric = false;
        try {
            std::stod(fields[0]);
            first_is_numeric = true;
        } catch (...) {}

        if (!first_is_numeric) {
            close_col = find_close_column(line);
            continue;
        }

        // Data row: extract the price.
        double price = 0.0;
        bool   got   = false;

        if (close_col >= 0 && close_col < static_cast<int>(fields.size())) {
            try { price = std::stod(fields[close_col]); got = (price > 0); }
            catch (...) {}
        } else if (fields.size() >= 2) {
            // date,price format: use second field
            try { price = std::stod(fields[1]); got = (price > 0); }
            catch (...) {}
        } else {
            // single-column
            try { price = std::stod(fields[0]); got = (price > 0); }
            catch (...) {}
        }

        if (got) prices.push_back(price);
    }

    if (prices.size() < 30)
        throw std::runtime_error(
            "Too few price observations (" + std::to_string(prices.size()) +
            "). Need at least 30 for meaningful parameter estimation.");

    return prices;
}

// Estimate GBM and jump-diffusion parameters from a price series.
//
// GBM parameters (mu, sigma) come from the sample mean and standard
// deviation of daily log returns, scaled to annual by ×252 and ×√252.
//
// Jump parameters (lambda, mu_j, sigma_j) use a threshold heuristic:
// returns beyond mean ± 2.5σ are classified as jumps. This is not a
// formal statistical test — it's a practical starting point. The
// threshold choice affects lambda materially; MLE would be more rigorous
// but adds significant complexity for modest improvement in this context.
ParamEstimates estimate_params(const std::vector<double>& prices) {
    const int n = static_cast<int>(prices.size());

    // Log returns
    std::vector<double> returns(n - 1);
    for (int i = 0; i < n - 1; ++i)
        returns[i] = std::log(prices[i + 1] / prices[i]);

    const double mean_r = [&] {
        double s = 0; for (double r : returns) s += r; return s / returns.size();
    }();

    const double var_r = [&] {
        double s = 0;
        for (double r : returns) s += (r - mean_r) * (r - mean_r);
        return s / (returns.size() - 1);  // Bessel's correction
    }();
    const double std_r = std::sqrt(var_r);

    // Annualise: 252 trading days per year
    const double mu    = mean_r * 252;
    const double sigma = std_r  * std::sqrt(252.0);

    // Jump detection: classify returns beyond 2.5 standard deviations as jumps.
    // Separate the jump returns from the diffusion returns for parameter estimation.
    const double threshold = 2.5 * std_r;
    std::vector<double> jump_returns;
    for (double r : returns) {
        if (std::abs(r - mean_r) > threshold)
            jump_returns.push_back(r);
    }

    double lambda = 0, mu_j = 0, sigma_j = 0;
    if (!jump_returns.empty()) {
        const double n_years = static_cast<double>(returns.size()) / 252.0;
        lambda = jump_returns.size() / n_years;

        mu_j = [&] {
            double s = 0; for (double r : jump_returns) s += r; return s / jump_returns.size();
        }();
        sigma_j = [&] {
            double s = 0;
            for (double r : jump_returns) s += (r - mu_j) * (r - mu_j);
            return std::sqrt(s / jump_returns.size());
        }();
    } else {
        // No jumps detected — use conservative defaults
        lambda  = 1.0;
        mu_j    = -0.03;
        sigma_j = 0.05;
    }

    return ParamEstimates{
        prices.back(),
        mu, sigma,
        lambda, mu_j, sigma_j,
        n,
        static_cast<int>(jump_returns.size())
    };
}

// Higher moments of the empirical return distribution.
// Used to check whether the simulated distribution matches what actually
// happened in the data — the key diagnostic for model adequacy.
struct EmpiricalMoments {
    double skewness;        // typically negative for equities (crash skew)
    double excess_kurtosis; // typically 3-10 for daily equity returns (fat tails)
    int    n_returns;
};

// Compute skewness and excess kurtosis of daily log returns from a price series.
// The excess kurtosis is the most direct measure of whether log-normal (GBM)
// is adequate: a value significantly above 0 means GBM will underestimate tail risk.
inline EmpiricalMoments empirical_moments(const std::vector<double>& prices) {
    const int n = static_cast<int>(prices.size());
    std::vector<double> returns(n - 1);
    for (int i = 0; i < n - 1; ++i)
        returns[i] = std::log(prices[i + 1] / prices[i]);

    const int m = static_cast<int>(returns.size());
    double mean = 0.0;
    for (double r : returns) mean += r;
    mean /= m;

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (double r : returns) {
        const double d  = r - mean;
        const double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= m; m3 /= m; m4 /= m;
    const double std_dev = std::sqrt(m2);

    const double skewness       = (std_dev > 0) ? m3 / (m2 * std_dev) : 0.0;
    const double excess_kurtosis = (m2 > 0)     ? m4 / (m2 * m2) - 3.0 : 0.0;

    return EmpiricalMoments{skewness, excess_kurtosis, m};
}
