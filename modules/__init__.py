# modules/__init__.py - Makes the modules directory a Python package

"""
Crypto BB Bounce Scanner - Modular Components

This package contains all the modular components for the BB bounce scanner:
- data_fetcher: Market data collection and exchange management
- bb_detector: Core Bollinger Band bounce detection logic
- technical_analyzer: Divergence detection and technical confirmations
- sentiment_analyzer: LunarCrush and TokenMetrics API integration
- risk_manager: Probability calculation and risk assessment
- output_generator: Excel generation and terminal display

Usage:
    from modules.data_fetcher import MarketDataFetcher
    from modules.bb_detector import BBDetector
    # etc.
"""

__version__ = "1.0.0"
__author__ = "Crypto BB Scanner Team"