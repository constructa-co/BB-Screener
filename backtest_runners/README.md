# Backtest Runners

This directory contains runner scripts for various backtest strategies.

## 📊 Volume Profile Backtests

### `run_r9_backtest.py`
- **Strategy:** Volume Profile Backtest (Revision 9)
- **Timeframe:** 4-hour charts
- **Purpose:** Tests volume profile trading strategies
- **Usage:** `python run_r9_backtest.py`

### `run_r10_backtest.py`
- **Strategy:** Volume Profile Backtest (Revision 10)
- **Timeframe:** 4-hour charts
- **Purpose:** Enhanced version of R9 with improvements
- **Usage:** `python run_r10_backtest.py`

### `run_r11_backtest.py`
- **Strategy:** Volume Profile Backtest (Revision 11) - Quality-Focused Optimization
- **Timeframe:** 4-hour charts
- **Purpose:** Quality-focused optimization of volume profile strategy
- **Usage:** `python run_r11_backtest.py`
- **Test Symbols:** BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, BNBUSDT

## 🔧 Integration Scripts

### `integrate_bb_backtest.py`
- **Purpose:** Integrates backtest results with the main BB scanner
- **Usage:** `python integrate_bb_backtest.py`

### `run_scanner.py`
- **Purpose:** General scanner runner script
- **Usage:** `python run_scanner.py`

## 📈 Running Backtests

To run any backtest, navigate to this directory and execute:

```bash
cd backtest_runners
python run_r11_backtest.py  # or any other version
```

## 📁 Output

Results are saved to `backtest_results/volume_profile/` in the main project directory.
