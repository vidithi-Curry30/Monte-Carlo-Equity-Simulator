# Sample Data

Download historical closing prices from Yahoo Finance and place them here.

Suggested tickers for reproducing the findings in `FINDINGS.md`:

| File          | Ticker | Period            | Why                                      |
|---------------|--------|-------------------|------------------------------------------|
| `spy.csv`     | SPY    | 2019-01-01 to present | Baseline — diversified, low jump intensity |
| `nvda.csv`    | NVDA   | 2019-01-01 to present | High-vol tech — large CVaR gap expected  |
| `aapl.csv`    | AAPL   | 2019-01-01 to present | Mature tech — moderate gap               |

**Download steps (Yahoo Finance):**
1. Go to finance.yahoo.com → search ticker → Historical Data
2. Set date range, select "Close" prices, download CSV
3. Place the file here and run: `./build/simulator --csv data/spy.csv --years 1 --compare`
