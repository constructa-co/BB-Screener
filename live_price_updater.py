#!/usr/bin/env python3
"""
Live Price Updater for Dashboard
Fetches real-time prices and updates dashboard data
"""

import requests
import pandas as pd
from datetime import datetime
import time
import json

class LivePriceUpdater:
    def __init__(self):
        self.price_cache = {}
        self.last_update = {}
        self.cache_duration = 30  # 30 seconds cache
        
    def get_live_price(self, symbol):
        """Get live price for a symbol"""
        current_time = time.time()
        
        # Check cache
        if (symbol in self.price_cache and 
            symbol in self.last_update and 
            current_time - self.last_update[symbol] < self.cache_duration):
            return self.price_cache[symbol]
        
        try:
            # Use Binance API for live prices
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": symbol.replace("/", "")}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                
                # Cache the result
                self.price_cache[symbol] = price
                self.last_update[symbol] = current_time
                
                return price
            else:
                print(f"❌ Failed to get price for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching price for {symbol}: {e}")
            return None
    
    def get_live_prices_batch(self, symbols):
        """Get live prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_live_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    def update_opportunity_prices(self, opportunities):
        """Update opportunity data with live prices"""
        if not opportunities:
            return opportunities
        
        # Extract unique symbols
        symbols = list(set([opp.get('symbol', '') for opp in opportunities if opp.get('symbol')]))
        
        # Get live prices
        live_prices = self.get_live_prices_batch(symbols)
        
        # Update opportunities with live prices
        updated_opportunities = []
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            if symbol in live_prices:
                # Update with live price
                opp['current_price'] = live_prices[symbol]
                opp['price_updated'] = datetime.now()
                
                # Recalculate targets based on live price
                if opp.get('entry_price') and opp.get('risk_reward_ratio'):
                    entry = float(opp['entry_price'])
                    rr_ratio = float(opp['risk_reward_ratio'])
                    
                    # Calculate new targets based on live price
                    current_price = live_prices[symbol]
                    if current_price > entry:
                        # Price moved up, adjust targets
                        price_diff = current_price - entry
                        opp['entry_price'] = current_price
                        opp['target_1'] = current_price + (price_diff * rr_ratio)
                        opp['stop_loss'] = current_price - (price_diff * 0.5)
                    else:
                        # Price moved down, keep original targets
                        pass
                
            updated_opportunities.append(opp)
        
        return updated_opportunities

# Global instance
price_updater = LivePriceUpdater()

def get_live_price_for_symbol(symbol):
    """Get live price for a single symbol"""
    return price_updater.get_live_price(symbol)

def update_opportunities_with_live_prices(opportunities):
    """Update opportunities with live prices"""
    return price_updater.update_opportunity_prices(opportunities)

def get_live_prices_for_symbols(symbols):
    """Get live prices for multiple symbols"""
    return price_updater.get_live_prices_batch(symbols) 