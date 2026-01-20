#!/usr/bin/env python3
"""
Simple health monitor test runner
"""

import asyncio
import sys
import os

# Add paths
sys.path.append('/Users/alper/Desktop/GitHubProjeleri/vital/backend')
sys.path.append('/Users/alper/Desktop/GitHubProjeleri/vital')

async def test_health_monitor():
    """Test health monitor functionality."""
    try:
        from app.websocket.health_monitor import WebSocketHealthMonitor
        from app.websocket.connection_manager import EnhancedConnectionManager
        from unittest.mock import Mock, AsyncMock
        
        print("🧪 Starting Health Monitor Tests...")
        
        # Test 1: Initialization
        print("📋 Test 1: Initialization")
        manager = EnhancedConnectionManager()
        monitor = WebSocketHealthMonitor(manager)
        await monitor.start_monitoring()
        print("✅ Health monitor initialized successfully")
        
        # Test 2: Connection Monitoring
        print("📋 Test 2: Connection Monitoring")
        mock_ws = Mock()
        mock_ws.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_ws, "user1", "doctor", 123)
        await monitor.monitor_connection(connection_id)
        
        health = await monitor.get_connection_health(connection_id)
        assert health is not None
        assert health.connection_id == connection_id
        assert health.user_id == "user1"
        print("✅ Connection monitoring works")
        
        # Test 3: Heartbeat Recording
        print("📋 Test 3: Heartbeat Recording")
        await monitor.record_heartbeat(connection_id, latency=50.0)
        
        health = await monitor.get_connection_health(connection_id)
        assert health.latency_ms == 50.0
        assert health.missed_heartbeats == 0
        print("✅ Heartbeat recording works")
        
        # Test 4: Health Check
        print("📋 Test 4: Health Check")
        status = await monitor.check_connection_health(connection_id)
        assert status.value in ['healthy', 'warning', 'unhealthy', 'critical']
        print("✅ Health check works")
        
        # Test 5: Metrics
        print("📋 Test 5: Metrics Collection")
        metrics = await monitor.get_health_metrics()
        assert metrics.total_connections >= 1
        print(f"✅ Metrics collected: {metrics.total_connections} connections")
        
        # Cleanup
        await monitor.shutdown()
        await manager.disconnect(connection_id)
        print("✅ Cleanup completed")
        
        print("\n🎉 ALL HEALTH MONITOR TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_health_monitor())
    sys.exit(0 if success else 1)
