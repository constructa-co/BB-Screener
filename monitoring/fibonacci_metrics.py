"""
Fibonacci Scanner Prometheus Metrics Endpoint
Provides health and performance metrics for monitoring
"""

from flask import Flask, Response
import psycopg2
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'bb_screener'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    try:
        conn = get_db_connection()
        if not conn:
            return Response('fibonacci_scanner_up {scanner="fibonacci"} 0\n# Error: Database connection failed', mimetype='text/plain')
        
        cur = conn.cursor()
        
        # Collect metrics
        metrics_lines = []
        
        # Total signals
        cur.execute("SELECT COUNT(*) FROM other_scanners.fibonacci_signals")
        total_signals = cur.fetchone()[0]
        metrics_lines.append(f'fibonacci_total_signals {{scanner="fibonacci"}} {total_signals}')
        
        # Signals last hour
        cur.execute("""
            SELECT COUNT(*) FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
        """)
        hourly_signals = cur.fetchone()[0]
        metrics_lines.append(f'fibonacci_signals_per_hour {{scanner="fibonacci"}} {hourly_signals}')
        
        # Average confidence
        cur.execute("""
            SELECT AVG(confidence_score) FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
        """)
        avg_confidence = cur.fetchone()[0] or 0
        metrics_lines.append(f'fibonacci_avg_confidence {{scanner="fibonacci"}} {avg_confidence:.4f}')
        
        # High confidence ratio
        cur.execute("""
            SELECT 
                COUNT(CASE WHEN confidence_score >= 0.7 THEN 1 END)::float / NULLIF(COUNT(*), 0)
            FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
        """)
        high_conf_ratio = cur.fetchone()[0] or 0
        metrics_lines.append(f'fibonacci_high_confidence_ratio {{scanner="fibonacci"}} {high_conf_ratio:.4f}')
        
        # Average quality score
        cur.execute("""
            SELECT AVG(quality_score) FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
        """)
        avg_quality = cur.fetchone()[0] or 0
        metrics_lines.append(f'fibonacci_avg_quality_score {{scanner="fibonacci"}} {avg_quality:.2f}')
        
        # Signals by type
        cur.execute("""
            SELECT signal_type, COUNT(*) 
            FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
            GROUP BY signal_type
        """)
        for signal_type, count in cur.fetchall():
            metrics_lines.append(f'fibonacci_signals_by_type {{type="{signal_type}"}} {count}')
        
        # Enhanced trading data metrics
        cur.execute("""
            SELECT 
                AVG(move_percentage) as avg_move,
                AVG(risk_percentage) as avg_risk,
                COUNT(CASE WHEN target_1_risk_reward >= 1.0 THEN 1 END) as good_rr_signals,
                COUNT(CASE WHEN entry_timing_status = 'NOW' THEN 1 END) as immediate_entries,
                COUNT(CASE WHEN entry_timing_status = 'WAIT' THEN 1 END) as waiting_entries
            FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
        """)
        trading_metrics = cur.fetchone()
        if trading_metrics:
            avg_move, avg_risk, good_rr, immediate_entries, waiting_entries = trading_metrics
            metrics_lines.append(f'fibonacci_avg_move_percentage {{scanner="fibonacci"}} {avg_move or 0:.2f}')
            metrics_lines.append(f'fibonacci_avg_risk_percentage {{scanner="fibonacci"}} {avg_risk or 0:.2f}')
            metrics_lines.append(f'fibonacci_good_risk_reward_signals {{scanner="fibonacci"}} {good_rr or 0}')
            metrics_lines.append(f'fibonacci_immediate_entries {{scanner="fibonacci"}} {immediate_entries or 0}')
            metrics_lines.append(f'fibonacci_waiting_entries {{scanner="fibonacci"}} {waiting_entries or 0}')
        
        # Performance by Fibonacci level
        cur.execute("""
            SELECT 
                fibonacci_level,
                COUNT(*) as signal_count,
                AVG(confidence_score) as avg_confidence
            FROM other_scanners.fibonacci_signals 
            WHERE detected_at > NOW() - INTERVAL '1 hour'
            GROUP BY fibonacci_level
        """)
        for level, count, confidence in cur.fetchall():
            metrics_lines.append(f'fibonacci_signals_by_level {{level="{level}"}} {count}')
            metrics_lines.append(f'fibonacci_confidence_by_level {{level="{level}"}} {confidence or 0:.4f}')
        
        # Scanner health metrics
        cur.execute("""
            SELECT 
                MAX(detected_at) as last_signal_time
            FROM other_scanners.fibonacci_signals
        """)
        last_signal_time = cur.fetchone()[0]
        
        if last_signal_time:
            time_since_last = (datetime.now() - last_signal_time).total_seconds()
            metrics_lines.append(f'fibonacci_seconds_since_last_signal {{scanner="fibonacci"}} {time_since_last}')
        
        cur.close()
        conn.close()
        
        # Add health metric
        metrics_lines.append('fibonacci_scanner_up {scanner="fibonacci"} 1')
        
        return Response('\n'.join(metrics_lines), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        # Return error metric
        return Response(f'fibonacci_scanner_up {{scanner="fibonacci"}} 0\n# Error: {str(e)}', mimetype='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        if not conn:
            return {'status': 'unhealthy', 'scanner': 'fibonacci', 'error': 'Database connection failed'}, 503
        
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        
        return {
            'status': 'healthy', 
            'scanner': 'fibonacci',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }, 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy', 
            'scanner': 'fibonacci', 
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 503

@app.route('/')
def index():
    """Root endpoint with basic info"""
    return {
        'service': 'Fibonacci Scanner Metrics',
        'version': '1.0.0',
        'endpoints': {
            '/metrics': 'Prometheus metrics endpoint',
            '/health': 'Health check endpoint'
        },
        'timestamp': datetime.now().isoformat()
    }

if __name__ == '__main__':
    port = int(os.getenv('FIBONACCI_METRICS_PORT', 9092))
    host = os.getenv('FIBONACCI_METRICS_HOST', '0.0.0.0')
    
    logger.info(f"Starting Fibonacci metrics server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
