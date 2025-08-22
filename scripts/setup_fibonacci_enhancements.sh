#!/bin/bash

echo "🚀 Setting up Fibonacci Scanner Enhancements..."
echo "================================================"

# 1. Create monitoring directory
echo "📁 Creating monitoring directory..."
mkdir -p monitoring
mkdir -p logs

# 2. Install required packages
echo "📦 Installing dependencies..."
pip install flask streamlit plotly

# 3. Start monitoring endpoint
echo "📊 Starting Prometheus metrics endpoint..."
nohup python monitoring/fibonacci_metrics.py > logs/fibonacci_metrics.log 2>&1 &

# 4. Add to cron for auto-start
echo "⏰ Adding to system startup..."
(crontab -l 2>/dev/null; echo "@reboot cd $(pwd) && python monitoring/fibonacci_metrics.py > logs/fibonacci_metrics.log 2>&1") | crontab -

# 5. Create health check script
echo "🔍 Creating health check script..."
cat > scripts/check_fibonacci_enhancements.sh << 'EOF'
#!/bin/bash
echo "=== FIBONACCI ENHANCEMENTS HEALTH CHECK ==="
echo ""

# Check monitoring endpoint
echo "📊 Checking monitoring endpoint..."
if curl -s http://localhost:9092/health > /dev/null; then
    echo "✅ Monitoring endpoint is responding"
    curl -s http://localhost:9092/health | python3 -m json.tool 2>/dev/null || echo "Response received"
else
    echo "❌ Monitoring endpoint not responding"
fi

# Check metrics
echo ""
echo "📈 Checking metrics endpoint..."
curl -s http://localhost:9092/metrics | head -10

# Check dashboard
echo ""
echo "📐 Checking Streamlit dashboard..."
if pgrep -f "streamlit.*dashboard" > /dev/null; then
    echo "✅ Streamlit dashboard is running"
else
    echo "❌ Streamlit not running"
fi

# Check Fibonacci scanner
echo ""
echo "🎯 Checking Fibonacci scanner..."
if pgrep -f "fibonacci_retracement_scanner" > /dev/null; then
    echo "✅ Fibonacci scanner is running"
else
    echo "❌ Fibonacci scanner not running"
fi

echo ""
echo "✅ Health check complete"
EOF

chmod +x scripts/check_fibonacci_enhancements.sh

# 6. Create environment variables file
echo "🔧 Creating environment configuration..."
cat > .env.fibonacci << 'EOF'
# Fibonacci Scanner Enhancement Configuration
FIBONACCI_METRICS_PORT=9092
FIBONACCI_METRICS_HOST=0.0.0.0
FIBONACCI_ASYNC_MODE=false

# Database Configuration (update with your actual values)
DB_HOST=localhost
DB_NAME=bb_screener
DB_USER=your_db_user
DB_PASSWORD=your_db_password
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Access points:"
echo "  Dashboard: http://your-server:8501 (navigate to Fibonacci Analysis page)"
echo "  Metrics: http://your-server:9092/metrics"
echo "  Health: http://your-server:9092/health"
echo ""
echo "🔧 To enable async scanning, set environment variable:"
echo "   export FIBONACCI_ASYNC_MODE=true"
echo ""
echo "📋 To check everything is working:"
echo "   ./scripts/check_fibonacci_enhancements.sh"
echo ""
echo "📊 To view metrics in real-time:"
echo "   watch -n 5 'curl -s http://localhost:9092/metrics | head -20'"
echo ""
echo "🔄 To restart monitoring:"
echo "   pkill -f fibonacci_metrics && nohup python monitoring/fibonacci_metrics.py > logs/fibonacci_metrics.log 2>&1 &"
