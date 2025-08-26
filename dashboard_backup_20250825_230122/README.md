# Dashboard Module

This directory contains dashboard-related files for the BB Scanner.

## 📊 Files

### `dashboard.py`
- **Purpose:** Main dashboard application
- **Usage:** `python dashboard.py`
- **Description:** Interactive dashboard for viewing BB Scanner results and analytics

### `debug_dashboard_data.py`
- **Purpose:** Debug dashboard data issues
- **Usage:** `python debug_dashboard_data.py`
- **Description:** Troubleshooting tool for dashboard data problems

### `interactive_controls.py`
- **Purpose:** Interactive controls for the dashboard
- **Usage:** Imported by dashboard.py
- **Description:** Provides interactive elements and controls for the dashboard interface

### `file_management.py`
- **Purpose:** File management utilities for dashboard
- **Usage:** Imported by dashboard components
- **Description:** Handles file operations, Excel generation, and data export

### `live_price_updater.py`
- **Purpose:** Live price updates for dashboard
- **Usage:** Imported by dashboard components
- **Description:** Provides real-time price data for dashboard displays

### `tradingview_charts.py`
- **Purpose:** TradingView chart integration
- **Usage:** Imported by dashboard components
- **Description:** Generates TradingView chart links and embeds

## 🚀 Running the Dashboard

```bash
cd dashboard
python dashboard.py
```

## 🔧 Features

- **Real-time data display**
- **Interactive charts**
- **Trade result visualization**
- **Performance analytics**
- **Export capabilities**

## 📈 Integration

The dashboard integrates with:
- Main scanner results
- Database data
- Live price feeds
- TradingView charts
