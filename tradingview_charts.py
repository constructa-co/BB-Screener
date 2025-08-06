"""
TradingView Chart Integration for Streamlit Dashboard
"""

import streamlit.components.v1 as components

def show_tradingview_chart(symbol, timeframe='240', height=600, studies=None):
    """
    Embed an interactive TradingView chart in Streamlit
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Chart interval - '1', '5', '15', '60', '240', 'D', 'W'
        height: Chart height in pixels
        studies: List of indicators to add
    """
    # Convert symbol format for TradingView
    tv_symbol = symbol.replace('/', '').replace('-', '')
    if 'USDT' in tv_symbol:
        exchange = 'BINANCE'
    else:
        exchange = 'BINANCE'  # Default exchange
    
    # Default studies if none provided
    if studies is None:
        studies = [
            "BB@tv-basicstudies",  # Bollinger Bands
            "RSI@tv-basicstudies",  # RSI
            "MACD@tv-basicstudies"  # MACD
        ]
    
    # Convert timeframe to TradingView format
    timeframe_map = {
        '1m': '1',
        '5m': '5', 
        '15m': '15',
        '1h': '60',
        '4h': '240',
        'D': 'D',
        'W': 'W'
    }
    
    tv_interval = timeframe_map.get(timeframe, timeframe)
    
    tradingview_widget = f"""
    <div class="tradingview-widget-container" style="width: 100%; height: {height}px; margin: 0; padding: 0;">
        <div id="tradingview_{tv_symbol}" style="width: 100%; height: 100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "autosize": true,
            "symbol": "{exchange}:{tv_symbol}",
            "interval": "{tv_interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_{tv_symbol}",
            "hide_side_toolbar": false,
            "studies": {studies},
            "show_popup_button": true,
            "popup_width": "1000",
            "popup_height": "650"
        }});
        </script>
    </div>
    """
    
    components.html(tradingview_widget, height=height)

def show_mini_chart(symbol, width=350, height=200):
    """
    Show a mini chart for quick preview
    """
    tv_symbol = symbol.replace('/', '').replace('-', '')
    
    mini_chart = f"""
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
        {{
            "symbol": "BINANCE:{tv_symbol}",
            "width": {width},
            "height": {height},
            "locale": "en",
            "dateRange": "1D",
            "colorTheme": "dark",
            "trendLineColor": "rgba(41, 98, 255, 1)",
            "underLineColor": "rgba(41, 98, 255, 0.3)",
            "underLineBottomColor": "rgba(41, 98, 255, 0)",
            "isTransparent": false,
            "autosize": false,
            "largeChartUrl": ""
        }}
        </script>
    </div>
    """
    
    components.html(mini_chart, height=height)

def show_technical_analysis_widget(symbol):
    """
    Show TradingView's technical analysis summary widget
    """
    tv_symbol = symbol.replace('/', '').replace('-', '')
    
    ta_widget = f"""
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
        {{
            "interval": "4h",
            "width": "100%",
            "isTransparent": false,
            "height": 450,
            "symbol": "BINANCE:{tv_symbol}",
            "showIntervalTabs": true,
            "locale": "en",
            "colorTheme": "dark"
        }}
        </script>
    </div>
    """
    
    components.html(ta_widget, height=450) 