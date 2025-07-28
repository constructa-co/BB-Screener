# modules/market_metadata_enricher.py
"""
Market Metadata Enricher - Standalone Module
Enhances trading data with market cap, liquidity, and sector information from CMC API
"""

import logging
import requests
from typing import Dict, Any, List, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketMetadataEnricher:
    """Enriches crypto data with market cap, liquidity, and sector metadata"""
    
    def __init__(self, cmc_api_key: Optional[str] = None):
        """
        Initialize the enricher
        
        Args:
            cmc_api_key: CoinMarketCap API key. If None, will try to import from config
        """
        if cmc_api_key:
            self.cmc_api_key = cmc_api_key
        else:
            try:
                from config import CMC_API_KEY
                self.cmc_api_key = CMC_API_KEY
            except ImportError:
                logger.warning("No CMC API key provided and config.py not found")
                self.cmc_api_key = None
                
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.cache = {}  # Simple cache to avoid repeated API calls
        self.request_count = 0  # Track API usage
        
        # Market cap tier definitions (matches your historical analysis)
        self.market_cap_tiers = {
            'large_cap': (1, 50),      # Top 50 - 63.7% success rate
            'mid_cap': (51, 150),      # 51-150 - Mixed performance  
            'small_cap': (151, 300),   # 151-300 - 74.6% success rate
            'micro_cap': (301, 1000)   # 301+ - Variable performance
        }
        
        # Liquidity tiers (24h volume based)
        self.liquidity_tiers = {
            'high_liquidity': 50_000_000,     # $50M+ daily volume
            'medium_liquidity': 10_000_000,   # $10M+ daily volume  
            'low_liquidity': 1_000_000,       # $1M+ daily volume
            'very_low_liquidity': 0            # Below $1M
        }
        
        logger.info(f"✅ MarketMetadataEnricher initialized. API Key: {'Available' if self.cmc_api_key else 'Missing'}")
        
    def enrich_trade_data(self, symbol: str, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich existing trade data with market metadata
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            existing_data: Current trade analysis data
            
        Returns:
            Enhanced data with market metadata
        """
        try:
            # Get metadata from CMC
            metadata = self._get_coin_metadata(symbol)
            
            # Enrich the existing data
            enhanced_data = existing_data.copy()
            enhanced_data.update({
                # Market Cap Intelligence
                'market_cap_usd': metadata.get('market_cap', 0),
                'market_cap_rank': metadata.get('cmc_rank', 999),
                'market_cap_tier': self._classify_market_cap_tier(metadata.get('cmc_rank', 999)),
                
                # Liquidity Intelligence  
                'volume_24h_usd': metadata.get('volume_24h', 0),
                'liquidity_tier': self._classify_liquidity_tier(metadata.get('volume_24h', 0)),
                'liquidity_risk_score': self._calculate_liquidity_risk(metadata.get('volume_24h', 0)),
                
                # Sector Intelligence
                'primary_sector': self._extract_primary_sector(metadata.get('tags', [])),
                'sector_category': self._classify_sector_category(metadata.get('tags', [])),
                'platform_ecosystem': metadata.get('platform', {}).get('name', 'Native'),
                
                # ML Features
                'is_large_cap': metadata.get('cmc_rank', 999) <= 50,
                'is_high_liquidity': metadata.get('volume_24h', 0) >= 50_000_000,
                'expected_success_rate': self._estimate_success_rate(
                    metadata.get('cmc_rank', 999), 
                    metadata.get('volume_24h', 0)
                ),
                
                # Risk Factors
                'liquidity_multiplier': self._calculate_liquidity_multiplier(metadata.get('volume_24h', 0)),
                'market_cap_multiplier': self._calculate_market_cap_multiplier(metadata.get('cmc_rank', 999))
            })
            
            logger.info(f"✅ Enhanced {symbol}: {enhanced_data['market_cap_tier']} cap, {enhanced_data['liquidity_tier']} liquidity, {enhanced_data['primary_sector']} sector")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Error enriching {symbol}: {e}")
            # Return original data with default metadata
            return self._add_default_metadata(existing_data)
    
    def _get_coin_metadata(self, symbol: str) -> Dict[str, Any]:
        """Fetch coin metadata from CMC API with caching"""
        
        # Check cache first
        if symbol in self.cache:
            logger.debug(f"📋 Using cached data for {symbol}")
            return self.cache[symbol]
            
        # Check if API key is available
        if not self.cmc_api_key:
            logger.warning(f"⚠️ No CMC API key available, using default metadata for {symbol}")
            return self._get_default_metadata()
            
        try:
            # CMC Quotes Latest API (includes market cap, volume, rank)
            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key}
            params = {"symbol": symbol, "convert": "USD"}
            
            logger.debug(f"🌐 Fetching CMC data for {symbol}...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            self.request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if symbol in data.get('data', {}):
                    coin_data = data['data'][symbol]
                    quote_data = coin_data.get('quote', {}).get('USD', {})
                    
                    metadata = {
                        'cmc_rank': coin_data.get('cmc_rank', 999),
                        'market_cap': quote_data.get('market_cap', 0),
                        'volume_24h': quote_data.get('volume_24h', 0),
                        'tags': coin_data.get('tags', []),
                        'category': coin_data.get('category', 'unknown'),
                        'platform': coin_data.get('platform') or {}
                    }
                    
                    # Cache the result
                    self.cache[symbol] = metadata
                    logger.debug(f"✅ CMC data cached for {symbol}")
                    return metadata
                else:
                    logger.warning(f"⚠️ {symbol} not found in CMC response")
                    
            else:
                logger.error(f"❌ CMC API error {response.status_code} for {symbol}")
                
            return self._get_default_metadata()
            
        except Exception as e:
            logger.error(f"❌ CMC API exception for {symbol}: {e}")
            return self._get_default_metadata()
    
    def _classify_market_cap_tier(self, rank: int) -> str:
        """Classify market cap tier based on CMC rank"""
        for tier, (min_rank, max_rank) in self.market_cap_tiers.items():
            if min_rank <= rank <= max_rank:
                return tier
        return 'micro_cap'
    
    def _classify_liquidity_tier(self, volume_24h: float) -> str:
        """Classify liquidity tier based on 24h volume"""
        if volume_24h >= self.liquidity_tiers['high_liquidity']:
            return 'high_liquidity'
        elif volume_24h >= self.liquidity_tiers['medium_liquidity']:
            return 'medium_liquidity'
        elif volume_24h >= self.liquidity_tiers['low_liquidity']:
            return 'low_liquidity'
        else:
            return 'very_low_liquidity'
    
    def _calculate_liquidity_risk(self, volume_24h: float) -> float:
        """Calculate liquidity risk score (0-1, higher = more risk)"""
        if volume_24h >= 50_000_000:  # $50M+
            return 0.1  # Very low risk
        elif volume_24h >= 10_000_000:  # $10M+
            return 0.3  # Low risk
        elif volume_24h >= 1_000_000:   # $1M+
            return 0.6  # Medium risk
        else:
            return 0.9  # High risk
    
    def _extract_primary_sector(self, tags: List[str]) -> str:
        """Extract primary sector from CMC tags"""
        
        # Sector mapping based on CMC tags
        sector_keywords = {
            'defi': ['defi', 'decentralized-finance', 'yield-farming', 'liquidity-mining'],
            'layer1': ['layer-1', 'smart-contracts', 'platform', 'blockchain'],
            'layer2': ['layer-2', 'scaling', 'ethereum-ecosystem', 'optimism-ecosystem'],
            'gaming': ['gaming', 'play-to-earn', 'metaverse', 'nft'],
            'meme': ['memes', 'dog-themed', 'meme-token'],
            'exchange': ['exchange-based', 'centralized-exchange', 'dex'],
            'privacy': ['privacy', 'privacy-coins', 'anonymous'],
            'oracle': ['oracles', 'oracle-token'],
            'storage': ['storage', 'file-storage', 'distributed-storage'],
            'infrastructure': ['infrastructure', 'web3', 'interoperability']
        }
        
        # Check tags against sector keywords
        for sector, keywords in sector_keywords.items():
            if any(keyword in tags for keyword in keywords):
                return sector
                
        return 'other'
    
    def _classify_sector_category(self, tags: List[str]) -> str:
        """High-level sector category for ML"""
        primary = self._extract_primary_sector(tags)
        
        # Group sectors into broader categories
        if primary in ['defi', 'exchange']:
            return 'financial'
        elif primary in ['layer1', 'layer2', 'infrastructure']:
            return 'infrastructure'
        elif primary in ['gaming', 'meme']:
            return 'speculative'
        else:
            return 'utility'
    
    def _estimate_success_rate(self, rank: int, volume_24h: float) -> float:
        """Estimate success rate based on historical performance"""
        
        # Base rates from your historical analysis
        if rank <= 50:  # Large cap
            base_rate = 63.7
        else:  # Smaller cap
            base_rate = 74.6
            
        # Liquidity adjustment
        if volume_24h >= 50_000_000:
            liquidity_bonus = 2.0  # High liquidity = slightly better
        elif volume_24h >= 1_000_000:
            liquidity_bonus = 0.0  # Normal
        else:
            liquidity_bonus = -5.0  # Low liquidity = worse performance
            
        return min(95.0, max(50.0, base_rate + liquidity_bonus))
    
    def _calculate_liquidity_multiplier(self, volume_24h: float) -> float:
        """Position size multiplier based on liquidity"""
        if volume_24h >= 50_000_000:   # High liquidity
            return 1.0
        elif volume_24h >= 10_000_000: # Medium liquidity  
            return 0.8
        elif volume_24h >= 1_000_000:  # Low liquidity
            return 0.5
        else:                          # Very low liquidity
            return 0.2
    
    def _calculate_market_cap_multiplier(self, rank: int) -> float:
        """Position size multiplier based on market cap performance"""
        if rank <= 50:        # Large cap (lower performance)
            return 0.8
        elif rank <= 300:     # Small cap (higher performance)
            return 1.2
        else:                 # Micro cap (variable)
            return 0.9
    
    def _get_default_metadata(self) -> Dict[str, Any]:
        """Default metadata when API fails"""
        return {
            'cmc_rank': 999,
            'market_cap': 0,
            'volume_24h': 0,
            'tags': [],
            'category': 'unknown',
            'platform': {}
        }
    
    def _add_default_metadata(self, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add default metadata to existing data"""
        enhanced_data = existing_data.copy()
        enhanced_data.update({
            'market_cap_usd': 0,
            'market_cap_rank': 999,
            'market_cap_tier': 'unknown',
            'volume_24h_usd': 0,
            'liquidity_tier': 'unknown',
            'liquidity_risk_score': 0.5,
            'primary_sector': 'unknown',
            'sector_category': 'unknown',
            'platform_ecosystem': 'unknown',
            'is_large_cap': False,
            'is_high_liquidity': False,
            'expected_success_rate': 72.4,  # Market average
            'liquidity_multiplier': 1.0,
            'market_cap_multiplier': 1.0
        })
        return enhanced_data
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache and API usage statistics"""
        return {
            'cached_symbols': len(self.cache),
            'api_requests_made': self.request_count
        }

# Example usage and helper functions
def enhance_trade_with_metadata(trade_result: Dict[str, Any], enricher: MarketMetadataEnricher) -> Dict[str, Any]:
    """Helper function to enhance a single trade result"""
    symbol = trade_result.get('symbol', 'UNKNOWN')
    return enricher.enrich_trade_data(symbol, trade_result)

def batch_enhance_trades(trade_results: List[Dict[str, Any]], enricher: MarketMetadataEnricher) -> List[Dict[str, Any]]:
    """Helper function to enhance a batch of trade results"""
    enhanced_results = []
    for trade in trade_results:
        enhanced_trade = enhance_trade_with_metadata(trade, enricher)
        enhanced_results.append(enhanced_trade)
    return enhanced_results