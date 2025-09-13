#!/usr/bin/env python3
"""
Process new signals from the signals table and add them to pending_signals
This will make the Signal Monitor pick up the new signals
"""

import sqlite3
import json
from datetime import datetime, timedelta

def process_new_signals():
    conn = sqlite3.connect('ict_trading.db')
    cursor = conn.cursor()
    
    # Get signals from the last hour that haven't been processed
    cursor.execute("""
        SELECT s.symbol, s.signal_type, s.quality_score, s.metadata, s.created_at
        FROM signals s
        LEFT JOIN pending_signals p ON s.symbol = p.symbol AND s.created_at = p.added_time
        WHERE s.created_at > datetime('now', '-1 hour')
        AND p.symbol IS NULL
        AND s.quality_score >= 70
        ORDER BY s.created_at DESC
        LIMIT 50
    """)
    
    new_signals = cursor.fetchall()
    print(f"Found {len(new_signals)} new signals to process")
    
    for signal in new_signals:
        symbol, signal_type, quality_score, metadata_str, created_at = signal
        
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            # Extract FVG data from metadata
            fvg_zone = metadata.get('fvg_zone', {})
            fvg_price = fvg_zone.get('entry_midpoint', 0.0)
            
            # Extract stop loss and take profit
            stop_loss = metadata.get('stop_loss', 0.0)
            take_profit = metadata.get('take_profit_1', 0.0)
            
            # Determine direction from side
            side = metadata.get('side', 'buy')
            direction = 'LONG' if side == 'buy' else 'SHORT'
            
            # Add to pending_signals
            expires_at = datetime.now() + timedelta(hours=24)  # Expire in 24 hours
            
            cursor.execute("""
                INSERT OR REPLACE INTO pending_signals 
                (signal_id, symbol, direction, fvg_price, stop_loss, take_profit, quality_score, added_time, expires_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                f"PROCESSED_{symbol}_{int(datetime.now().timestamp())}",
                symbol,
                direction,
                fvg_price,
                stop_loss,
                take_profit,
                quality_score,
                created_at,
                expires_at
            ))
            
            print(f"✅ Added {symbol} ({direction}) - Quality: {quality_score}, FVG: {fvg_price}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Processed {len(new_signals)} new signals")

if __name__ == "__main__":
    process_new_signals()
