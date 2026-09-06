# Liquidity Proxy & GBM Calibration

Pipeline to (1) download one day of 1-minute bars for a liquid and an illiquid
ticker, (2) compare high-minus-low per bar as a tightness/spread proxy, and
(3) calibrate a GBM from realized returns, simulate 10,000 paths, and verify
the mean/median identity.

**Note on execution:** this code uses `yfinance`, which requires internet
access to Yahoo Finance. It must be run on your own machine (or any
environment with unrestricted network access) — it will not run inside this
chat's sandbox, which only reaches package repositories.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Choosing tickers

Defaults are set in `src/config.py`:

```python
TICKER_LIQUID = "SPY"      # high ADV, tight quotes
TICKER_ILLIQUID = "GEVO"   # placeholder — swap for whatever illiquid name you want
```

`yfinance` only serves 1-minute bars for the **last 7 calendar days**, so
there's no need to pick a specific historical date — `period="1d"` always
grabs the most recent completed/partial session. Before committing to an
"illiquid" ticker, sanity-check its average daily volume:

```python
import yfinance as yf
yf.Ticker("GEVO").info.get("averageVolume")
```

Pick something with ADV well below SPY's (ideally under a few hundred
thousand shares/day, or a small/micro-cap with visibly wide bars) so the
high-minus-low contrast is meaningful. Edit `src/config.py` and re-run.

## Running the full pipeline

```bash
python main.py
```

This will:
1. Download 1-min OHLCV bars for both tickers → `data/raw/`
2. Compute high-minus-low (absolute and as % of close) per bar for each
   ticker, plot overlaid distributions → `outputs/hl_distribution.png`
3. Calibrate GBM drift/vol from the liquid ticker's realized 1-min log
   returns, simulate 10,000 terminal-price paths, and print:
   - theoretical mean `S0 * exp(mu*T)` vs simulated mean
   - theoretical median `S0 * exp((mu - sigma^2/2)*T)` vs simulated median
   - relative errors for both

## Tests

Unit tests use synthetic data and a mocked `yfinance.download`, so they run
with **no network access**:

```bash
pytest tests/ -v
```

## Analysis (fill in after running on real data)

> **Which BSM assumption did your simulation violate most, and how would
> you detect it numerically?**
>
> _[Write your one-paragraph answer here once you've inspected the
> high-minus-low distributions and the realized-return series. Hints: BSM
> assumes (a) continuous trading with no bid-ask friction, (b) i.i.d.
> log-normal returns with constant volatility, (c) no jumps. The
> high-minus-low proxy for the illiquid name is the most direct evidence
> against (a) — wide, erratic H-L bars relative to price signal a real
> bid-ask spread / discrete-price friction that a frictionless GBM can't
] > produce. Numerically you'd detect this via: (i) autocorrelation of 1-min
> log returns (bid-ask bounce induces negative lag-1 autocorrelation that
> GBM's independent increments rule out), (ii) a Jarque-Bera or
> excess-kurtosis test on realized returns vs. the simulated GBM returns
> (fat tails / volatility clustering violate constant-sigma i.i.d.
> log-normality), and (iii) comparing realized vs. GBM-implied
> autocorrelation of squared returns (volatility clustering, which GBM has
> none of).]_
