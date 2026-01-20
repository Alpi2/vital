import asyncio
import sys
sys.path.insert(0, '/Users/alper/Desktop/GitHubProjeleri/vital/backend')
sys.path.insert(0, '/Users/alper/Desktop/GitHubProjeleri/vital')

async def test_metrics_only():
    try:
        print("🔍 Testing metrics collection only...")
        
        from app.websocket.connection_manager import EnhancedConnectionManager
        from app.websocket.health_monitor import WebSocketHealthMonitor
        from unittest.mock import Mock, AsyncMock
        
        # Setup
        manager = EnhancedConnectionManager()
        mock_ws = Mock()
        mock_ws.send_text = AsyncMock()
        
        connection_id = await manager.connect(mock_ws, 'test_user', 'doctor', 123)
        monitor = WebSocketHealthMonitor(manager)
        await monitor.start_monitoring()
        await monitor.monitor_connection(connection_id)
        
        print("✅ Setup completed")
        
        # Test metrics collection
        print("🔍 Testing get_health_metrics...")
        metrics = await monitor.get_health_metrics()
        print(f"✅ Metrics: {metrics.total_connections} connections")
        
        # Cleanup
        await monitor.shutdown()
        await manager.disconnect(connection_id)
        print("✅ Test 5 PASSED!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_metrics_only())
