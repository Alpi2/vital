"""
Professional Health Monitor Test Suite

Fixed version with proper error handling and debugging.
"""

import asyncio
import sys
import os
import logging
from unittest.mock import Mock, AsyncMock

# Setup paths
sys.path.insert(0, '/Users/alper/Desktop/GitHubProjeleri/vital/backend')
sys.path.insert(0, '/Users/alper/Desktop/GitHubProjeleri/vital')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthMonitorTestSuite:
    """Professional test suite for WebSocket Health Monitor."""
    
    def __init__(self):
        self.manager = None
        self.monitor = None
        self.connection_id = None
        self.mock_websocket = None
    
    async def setup(self):
        """Setup test environment."""
        try:
            from app.websocket.connection_manager import EnhancedConnectionManager
            from app.websocket.health_monitor import WebSocketHealthMonitor
            
            logger.info("🔧 Setting up test environment...")
            
            # Create connection manager
            self.manager = EnhancedConnectionManager()
            
            # Create mock websocket
            self.mock_websocket = Mock()
            self.mock_websocket.send_text = AsyncMock()
            self.mock_websocket.close = AsyncMock()
            
            # Create health monitor
            self.monitor = WebSocketHealthMonitor(self.manager)
            await self.monitor.start_monitoring()
            
            logger.info("✅ Setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    async def teardown(self):
        """Cleanup test environment."""
        try:
            logger.info("🧹 Cleaning up test environment...")
            
            if self.connection_id and self.manager:
                await self.manager.disconnect(self.connection_id)
            
            if self.monitor:
                await self.monitor.shutdown()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    async def test_connection_creation(self):
        """Test 1: Connection Creation."""
        logger.info("📋 Test 1: Connection Creation")
        
        try:
            self.connection_id = await self.manager.connect(
                self.mock_websocket, 
                "test_user", 
                "doctor", 
                123
            )
            
            assert self.connection_id is not None
            assert len(self.manager.connections) == 1
            
            logger.info(f"✅ Connection created: {self.connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection creation failed: {e}")
            return False
    
    async def test_health_monitoring(self):
        """Test 2: Health Monitoring."""
        logger.info("📋 Test 2: Health Monitoring")
        
        try:
            if not self.connection_id:
                raise ValueError("No connection ID available")
            
            await self.monitor.monitor_connection(self.connection_id)
            
            health = await self.monitor.get_connection_health(self.connection_id)
            assert health is not None
            assert health.connection_id == self.connection_id
            assert health.user_id == "test_user"
            assert health.role == "doctor"
            assert health.status.value == "healthy"
            
            logger.info(f"✅ Health monitoring works: {health.connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health monitoring failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_heartbeat_recording(self):
        """Test 3: Heartbeat Recording."""
        logger.info("📋 Test 3: Heartbeat Recording")
        
        try:
            if not self.connection_id:
                raise ValueError("No connection ID available")
            
            await self.monitor.record_heartbeat(self.connection_id, latency=75.5)
            
            health = await self.monitor.get_connection_health(self.connection_id)
            assert health.latency_ms == 75.5
            assert health.missed_heartbeats == 0
            assert health.is_healthy == True
            
            logger.info(f"✅ Heartbeat recorded: {health.latency_ms}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Heartbeat recording failed: {e}")
            return False
    
    async def test_health_status_check(self):
        """Test 4: Health Status Check."""
        logger.info("📋 Test 4: Health Status Check")
        
        try:
            if not self.connection_id:
                raise ValueError("No connection ID available")
            
            status = await self.monitor.check_connection_health(self.connection_id)
            assert status is not None
            assert hasattr(status, 'value')
            
            logger.info(f"✅ Health status: {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health status check failed: {e}")
            return False
    
    async def test_metrics_collection(self):
        """Test 5: Metrics Collection."""
        logger.info("📋 Test 5: Metrics Collection")
        
        try:
            metrics = await self.monitor.get_health_metrics()
            assert metrics is not None
            assert metrics.total_connections >= 1
            assert hasattr(metrics, 'healthy_connections')
            assert hasattr(metrics, 'average_latency_ms')
            
            logger.info(f"✅ Metrics collected: {metrics.total_connections} connections")
            return True
            
        except Exception as e:
            logger.error(f"❌ Metrics collection failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test 6: Error Handling."""
        logger.info("📋 Test 6: Error Handling")
        
        try:
            if not self.connection_id:
                raise ValueError("No connection ID available")
            
            health = await self.monitor.get_connection_health(self.connection_id)
            original_error_count = health.error_count
            
            health.record_error("Test error")
            assert health.error_count == original_error_count + 1
            assert health.last_error == "Test error"
            
            logger.info("✅ Error handling works")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests sequentially."""
        logger.info("🚀 Starting Health Monitor Test Suite")
        
        try:
            await self.setup()
            
            tests = [
                self.test_connection_creation,
                self.test_health_monitoring,
                self.test_heartbeat_recording,
                self.test_health_status_check,
                self.test_metrics_collection,
                self.test_error_handling
            ]
            
            passed = 0
            failed = 0
            
            for i, test in enumerate(tests, 1):
                try:
                    result = await test()
                    if result:
                        passed += 1
                        logger.info(f"✅ Test {i} PASSED")
                    else:
                        failed += 1
                        logger.error(f"❌ Test {i} FAILED")
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Test {i} CRASHED: {e}")
            
            logger.info(f"\n📊 Test Results: {passed} passed, {failed} failed")
            
            if failed == 0:
                logger.info("🎉 ALL TESTS PASSED!")
                return True
            else:
                logger.error(f"💥 {failed} TESTS FAILED!")
                return False
                
        finally:
            await self.teardown()


async def main():
    """Main test runner."""
    test_suite = HealthMonitorTestSuite()
    success = await test_suite.run_all_tests()
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test runner crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
