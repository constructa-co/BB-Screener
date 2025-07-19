# modules/market_metadata_enricher.py
"""
Minimal Market Metadata Enricher for Testing
"""

import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MarketMetadataEnricher:
    """Test version of Market Metadata Enricher"""
    
    def __init__(self, cmc_api_key: Optional[str] = None):
        self.cmc_api_key = cmc_api_key
        if not cmc_api_key:
            try:
                from config import CMC_API_KEY
                self.cmc_api_key = CMC_API_KEY
            except ImportError:
                self.cmc_api_key = None
        
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.cache = {}
        self.request_count = 0
        
        # Market cap tiers
        self.market_cap_tiers = {
            'large_cap': (1, 50),
            'mid_cap': (51, 150),
            'small_cap': (151, 300),
            'micro_cap': (301, 1000)
        }
        
        print(f"✅ MarketMetadataEnricher initialized. API Key: {'Available' if self.cmc_api_key else 'Missing'}")
    
    def enrich_trade_data(self, symbol: str, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich trade data with metadata"""
        try:
            metadata = self._get_coin_metadata(symbol)
            
            enhanced_data = existing_data.copy()
            enhanced_data.update({
                'market_cap_usd': metadata.get('market_cap', 0),
                'market_cap_rank': metadata.get('cmc_rank', 999),
                'market_cap_tier': self._classify_market_cap_tier(metadata.get('cmc_rank', 999)),
                'volume_24h_usd': metadata.get('volume_24h', 0),
                'liquidity_tier': self._classify_liquidity_tier(metadata.get('volume_24h', 0)),
                'liquidity_risk_score': self._calculate_liquidity_risk(metadata.get('volume_24h', 0)),
                'primary_sector': self._extract_primary_sector(metadata.get('tags', [])),
                'sector_category': self._classify_sector_category(metadata.get('tags', [])),
                'platform_ecosystem': metadata.get('platform', {}).get('name', 'Native'),
                'is_large_cap': metadata.get('cmc_rank', 999) <= 50,
                'is_high_liquidity': metadata.get('volume_24h', 0) >= 50_000_000,
                'expected_success_rate': self._estimate_success_rate(metadata.get('cmc_rank', 999), metadata.get('volume_24h', 0)),
                'liquidity_multiplier': self._calculate_liquidity_multiplier(metadata.get('volume_24h', 0)),
                'market_cap_multiplier': self._calculate_market_cap_multiplier(metadata.get('cmc_rank', 999))
            })
            
            print(f"✅ Enhanced {symbol}: {enhanced_data['market_cap_tier']} cap, {enhanced_data['liquidity_tier']} liquidity")
            return enhanced_data
            
        except Exception as e:
            print(f"❌ Error enriching {symbol}: {e}")
            return self._add_default_metadata(existing_data)
    
    def _get_coin_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get coin metadata from CMC API"""
        if symbol in self.cache:
            return self.cache[symbol]
        
        if not self.cmc_api_key:
            print(f"⚠️ No API key, using defaults for {symbol}")
            return self._get_default_metadata()
        
        try:
            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key}
            params = {"symbol": symbol, "convert": "USD"}
            
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
                    
                    self.cache[symbol] = metadata
                    return metadata
            
            return self._get_default_metadata()
            
        except Exception as e:
            print(f"❌ API error for {symbol}: {e}")
            return self._get_default_metadata()
    
    def _classify_market_cap_tier(self, rank: int) -> str:
        """Classify market cap tier"""
        for tier, (min_rank, max_rank) in self.market_cap_tiers.items():
            if min_rank <= rank <= max_rank:
                return tier
        return 'micro_cap'
    
    def _classify_liquidity_tier(self, volume_24h: float) -> str:
        """Classify liquidity tier"""
        if volume_24h >= 50_000_000:
            return 'high_liquidity'
        elif volume_24h >= 10_000_000:
            return 'medium_liquidity'
        elif volume_24h >= 1_000_000:
            return 'low_liquidity'
        else:
            return 'very_low_liquidity'
    
    def _calculate_liquidity_risk(self, volume_24h: float) -> float:
        """Calculate liquidity risk score"""
        if volume_24h >= 50_000_000:
            return 0.1
        elif volume_24h >= 10_000_000:
            return 0.3
        elif volume_24h >= 1_000_000:
            return 0.6
        else:
            return 0.9
    
    def _extract_primary_sector(self, tags: List[str]) -> str:
        """Extract primary sector from tags"""
        sector_keywords = {
            'defi': ['defi', 'decentralized-finance', 'yield-farming'],
            'layer1': ['layer-1', 'smart-contracts', 'platform'],
            'layer2': ['layer-2', 'scaling', 'ethereum-ecosystem'],
            'gaming': ['gaming', 'play-to-earn', 'metaverse'],
            'meme': ['memes', 'dog-themed', 'meme-token'],
            'exchange': ['exchange-based', 'centralized-exchange'],
            'oracle': ['oracles', 'oracle-token']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in tags for keyword in keywords):
                return sector
        return 'other'
    
    def _classify_sector_category(self, tags: List[str]) -> str:
        """Classify high-level sector category"""
        primary = self._extract_primary_sector(tags)
        if primary in ['defi', 'exchange']:
            return 'financial'
        elif primary in ['layer1', 'layer2']:
            return 'infrastructure'
        elif primary in ['gaming', 'meme']:
            return 'speculative'
        else:
            return 'utility'
    
    def _estimate_success_rate(self, rank: int, volume_24h: float) -> float:
        """Estimate success rate"""
        base_rate = 63.7 if rank <= 50 else 74.6
        liquidity_bonus = 2.0 if volume_24h >= 50_000_000 else 0.0 if volume_24h >= 1_000_000 else -5.0
        return min(95.0, max(50.0, base_rate + liquidity_bonus))
    
    def _calculate_liquidity_multiplier(self, volume_24h: float) -> float:
        """Calculate liquidity multiplier"""
        if volume_24h >= 50_000_000:
            return 1.0
        elif volume_24h >= 10_000_000:
            return 0.8
        elif volume_24h >= 1_000_000:
            return 0.5
        else:
            return 0.2
    
    def _calculate_market_cap_multiplier(self, rank: int) -> float:
        """Calculate market cap multiplier"""
        if rank <= 50:
            return 0.8
        elif rank <= 300:
            return 1.2
        else:
            return 0.9
    
    def _get_default_metadata(self) -> Dict[str, Any]:
        """Default metadata"""
        return {
            'cmc_rank': 999,
            'market_cap': 0,
            'volume_24h': 0,
            'tags': [],
            'category': 'unknown',
            'platform': {}
        }
    
    def _add_default_metadata(self, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add default metadata"""
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
            'expected_success_rate': 72.4,
            'liquidity_multiplier': 1.0,
            'market_cap_multiplier': 1.0
        })
        return enhanced_data