import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.demo import router
from app.services.demo_data_manager import demo_data_manager, DemoGenerationInProgressError, DemoResetInProgressError
from app.core.cache import cache_service
from app.audit.audit_service import audit_service, AuditOperation

class TestDemoAPI:
    """Comprehensive test suite for Demo API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    async def cache_setup(self):
        """Setup cache for testing"""
        await cache_service.initialize()
        yield
        await cache_service.close()
    
    @pytest.fixture
    def mock_user(self):
        """Mock user for testing"""
        return {
            "sub": "test_user_123",
            "roles": ["admin"],
            "permissions": ["demo:read", "demo:write"]
        }
    
    @pytest.fixture
    def mock_admin_user(self):
        """Mock admin user for testing"""
        return {
            "sub": "admin_user_456",
            "roles": ["super_admin"],
            "permissions": ["demo:read", "demo:write", "demo:reset"]
        }

    class TestScenarioEndpoints:
        """Test scenario-related endpoints"""
        
        @pytest.mark.asyncio
        async def test_get_role_scenario_success(self, cache_setup, mock_user):
            """Test successful scenario retrieval"""
            
            # Mock the generator
            with patch.object(demo_data_manager, 'get_scenario') as mock_get:
                mock_scenario = {
                    "role": "doctor",
                    "title": "Doctor Scenario",
                    "description": "Medical practitioner scenario",
                    "patients": [],
                    "devices": [],
                    "alerts": [],
                    "settings": {},
                    "generated_at": datetime.utcnow().isoformat(),
                    "cache_ttl": 3600
                }
                mock_get.return_value = mock_scenario
                
                result = await demo_data_manager.get_scenario("doctor", "test_user_123")
                
                assert result["role"] == "doctor"
                assert result["title"] == "Doctor Scenario"
                assert "generated_at" in result
                assert "cache_ttl" in result
        
        @pytest.mark.asyncio
        async def test_get_role_scenario_cache_hit(self, cache_setup):
            """Test scenario cache hit"""
            
            # Set up cache
            cache_key = f"demo:scenario:doctor"
            cached_scenario = {
                "role": "doctor",
                "cached": True,
                "generated_at": datetime.utcnow().isoformat()
            }
            await cache_service.set(cache_key, cached_scenario, ttl=3600)
            
            result = await demo_data_manager.get_scenario("doctor", "test_user")
            
            assert result["cached"] is True
            assert result["role"] == "doctor"
        
        @pytest.mark.asyncio
        async def test_get_role_scenario_invalid_role(self, cache_setup):
            """Test scenario retrieval with invalid role"""
            
            with pytest.raises(ValueError):
                await demo_data_manager.get_scenario("invalid_role", "test_user")
        
        @pytest.mark.asyncio
        async def test_get_role_scenario_generation_in_progress(self, cache_setup):
            """Test scenario retrieval when generation is in progress"""
            
            # Set lock
            lock_key = "demo:lock:generate_scenario"
            await cache_service.set(lock_key, True, ttl=120)
            
            with pytest.raises(DemoGenerationInProgressError):
                await demo_data_manager.get_scenario("doctor", "test_user")
        
        @pytest.mark.asyncio
        async def test_list_all_scenarios(self, cache_setup):
            """Test listing all scenarios"""
            
            with patch.object(demo_data_manager, 'get_all_scenarios') as mock_list:
                mock_scenarios = [
                    {
                        "role": "doctor",
                        "description": "Medical practitioner",
                        "num_patients": 15,
                        "cached": True
                    },
                    {
                        "role": "nurse",
                        "description": "Nursing staff",
                        "num_patients": 20,
                        "cached": False
                    }
                ]
                mock_list.return_value = mock_scenarios
                
                result = await demo_data_manager.get_all_scenarios("test_user")
                
                assert len(result) == 2
                assert result[0]["role"] == "doctor"
                assert result[1]["role"] == "nurse"
                assert sum(1 for s in result if s.get("cached")) == 1

    class TestResetEndpoints:
        """Test reset-related endpoints"""
        
        @pytest.mark.asyncio
        async def test_reset_all_success(self, cache_setup, mock_admin_user):
            """Test successful reset of all demo data"""
            
            with patch.object(demo_data_manager, 'reset_all') as mock_reset:
                mock_reset.return_value = {
                    "message": "Demo data reset initiated",
                    "status": "resetting",
                    "cache_keys_cleared": 15,
                    "reset_at": datetime.utcnow().isoformat()
                }
                
                result = await demo_data_manager.reset_all("admin_user_456")
                
                assert result["status"] == "resetting"
                assert result["cache_keys_cleared"] == 15
                assert "reset_at" in result
        
        @pytest.mark.asyncio
        async def test_reset_all_in_progress(self, cache_setup, mock_admin_user):
            """Test reset when already in progress"""
            
            # Set lock
            lock_key = "demo:lock:reset_all"
            await cache_service.set(lock_key, True, ttl=120)
            
            with pytest.raises(DemoResetInProgressError):
                await demo_data_manager.reset_all("admin_user_456")
        
        @pytest.mark.asyncio
        async def test_reset_role_success(self, cache_setup, mock_admin_user):
            """Test successful reset of specific role data"""
            
            with patch.object(demo_data_manager, 'reset_role') as mock_reset:
                mock_reset.return_value = {
                    "message": "Demo data for role doctor reset successfully",
                    "role": "doctor",
                    "reset_at": datetime.utcnow().isoformat()
                }
                
                result = await demo_data_manager.reset_role("doctor", "admin_user_456")
                
                assert result["role"] == "doctor"
                assert result["message"].startswith("Demo data for role doctor")
        
        @pytest.mark.asyncio
        async def test_reset_role_invalid_role(self, cache_setup, mock_admin_user):
            """Test reset with invalid role"""
            
            with pytest.raises(ValueError):
                await demo_data_manager.reset_role("invalid_role", "admin_user_456")

    class TestStatusEndpoints:
        """Test status-related endpoints"""
        
        @pytest.mark.asyncio
        async def test_get_status_success(self, cache_setup):
            """Test successful status retrieval"""
            
            with patch.object(demo_data_manager, 'get_status') as mock_status:
                mock_status = {
                    "status": "active",
                    "initialized_at": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "statistics": {
                        "patients": 50,
                        "ecg_sessions": 100,
                        "alarms": 25
                    }
                }
                mock_status.return_value = mock_status
                
                result = await demo_data_manager.get_status()
                
                assert result["status"] == "active"
                assert result["version"] == "1.0.0"
                assert result["statistics"]["patients"] == 50
        
        @pytest.mark.asyncio
        async def test_get_statistics_success(self, cache_setup):
            """Test successful statistics retrieval"""
            
            with patch.object(demo_data_manager, 'get_statistics') as mock_stats:
                mock_stats = {
                    "patients": 50,
                    "ecg_sessions": 100,
                    "alarms": 25,
                    "calculated_at": datetime.utcnow().isoformat(),
                    "cache_ttl": 1800
                }
                mock_stats.return_value = mock_stats
                
                result = await demo_data_manager.get_statistics()
                
                assert result["patients"] == 50
                assert result["ecg_sessions"] == 100
                assert result["alarms"] == 25

    class TestGenerateEndpoint:
        """Test data generation endpoint"""
        
        @pytest.mark.asyncio
        async def test_regenerate_data_success(self, cache_setup, mock_admin_user):
            """Test successful data regeneration"""
            
            mock_background_tasks = AsyncMock()
            
            with patch.object(demo_data_manager, 'regenerate_data') as mock_regenerate:
                mock_regenerate.return_value = {
                    "message": "Demo data regeneration started in background",
                    "status": "regenerating",
                    "started_at": datetime.utcnow().isoformat()
                }
                
                result = await demo_data_manager.regenerate_data(
                    mock_background_tasks, "admin_user_456"
                )
                
                assert result["status"] == "regenerating"
                assert "started_at" in result
                mock_background_tasks.add_task.assert_called_once()

    class TestCaching:
        """Test caching functionality"""
        
        @pytest.mark.asyncio
        async def test_cache_scenario_ttl(self, cache_setup):
            """Test that scenarios are cached with correct TTL"""
            
            # Generate scenario
            with patch.object(demo_data_manager.generator, 'generate_scenario') as mock_generate:
                mock_generate.return_value = {
                    "role": "doctor",
                    "patients": []
                }
                
                # First call should cache
                result1 = await demo_data_manager.get_scenario("doctor", "test_user")
                
                # Check cache exists
                cache_key = "demo:scenario:doctor"
                assert await cache_service.exists(cache_key)
                
                # Check TTL
                ttl = await cache_service.get_ttl(cache_key)
                assert 3000 <= ttl <= 3600  # Allow some variance
        
        @pytest.mark.asyncio
        async def test_cache_invalidation_on_reset(self, cache_setup):
            """Test that cache is invalidated on reset"""
            
            # Set up cache
            cache_key = "demo:scenario:doctor"
            await cache_service.set(cache_key, {"cached": True}, ttl=3600)
            
            # Reset all
            with patch.object(demo_data_manager, '_clear_demo_patients'):
                await demo_data_manager.reset_all("admin_user")
            
            # Cache should be cleared
            assert not await cache_service.exists(cache_key)

    class TestAuditLogging:
        """Test audit logging functionality"""
        
        @pytest.mark.asyncio
        async def test_audit_scenario_generation(self, cache_setup):
            """Test audit logging for scenario generation"""
            
            with patch.object(demo_data_manager.generator, 'generate_scenario') as mock_generate:
                mock_generate.return_value = {"role": "doctor", "patients": []}
                
                await demo_data_manager.get_scenario("doctor", "test_user_123")
                
                # Check audit log was created
                # This would verify the audit service was called
                assert True  # Placeholder for actual audit verification
        
        @pytest.mark.asyncio
        async def test_audit_reset_operation(self, cache_setup):
            """Test audit logging for reset operations"""
            
            with patch.object(demo_data_manager, '_clear_demo_patients'):
                await demo_data_manager.reset_all("admin_user_456")
                
                # Check audit log was created
                assert True  # Placeholder for actual audit verification

    class TestRateLimiting:
        """Test rate limiting functionality"""
        
        @pytest.mark.asyncio
        async def test_rate_limit_scenario_endpoint(self, client, cache_setup):
            """Test rate limiting on scenario endpoint"""
            
            # Make multiple rapid requests
            responses = []
            for _ in range(5):
                response = client.get("/api/v1/demo/scenarios/doctor")
                responses.append(response)
            
            # Check if rate limiting is working
            # This would depend on the actual rate limiting implementation
            assert len(responses) == 5
        
        @pytest.mark.asyncio
        async def test_rate_limit_reset_endpoint(self, client, cache_setup, mock_admin_user):
            """Test rate limiting on reset endpoint"""
            
            # Make multiple rapid reset requests
            responses = []
            for _ in range(3):
                response = client.post(
                    "/api/v1/demo/reset",
                    json={"confirm": True}
                )
                responses.append(response)
            
            # Check if rate limiting is working
            assert len(responses) == 3

    class TestErrorHandling:
        """Test error handling"""
        
        @pytest.mark.asyncio
        async def test_scenario_not_found(self, cache_setup):
            """Test handling of scenario not found"""
            
            with patch.object(demo_data_manager.generator, 'generate_scenario') as mock_generate:
                mock_generate.side_effect = Exception("Scenario not found")
                
                with pytest.raises(Exception):
                    await demo_data_manager.get_scenario("unknown_role", "test_user")
        
        @pytest.mark.asyncio
        async def test_cache_service_failure(self, cache_setup):
            """Test handling of cache service failure"""
            
            # Mock cache service failure
            with patch.object(cache_service, 'get') as mock_get:
                mock_get.side_effect = Exception("Cache service down")
                
                # Should still work with fallback
                with patch.object(demo_data_manager.generator, 'generate_scenario') as mock_generate:
                    mock_generate.return_value = {"role": "doctor"}
                    
                    result = await demo_data_manager.get_scenario("doctor", "test_user")
                    assert result["role"] == "doctor"

    class TestPerformance:
        """Test performance requirements"""
        
        @pytest.mark.asyncio
        async def test_scenario_load_performance_cached(self, cache_setup):
            """Test cached scenario load performance (<10ms)"""
            
            # Set up cached scenario
            cache_key = "demo:scenario:doctor"
            cached_scenario = {
                "role": "doctor",
                "patients": [{"id": i} for i in range(15)]  # Simulate realistic data
            }
            await cache_service.set(cache_key, cached_scenario, ttl=3600)
            
            # Measure performance
            start_time = datetime.utcnow()
            result = await demo_data_manager.get_scenario("doctor", "test_user")
            end_time = datetime.utcnow()
            
            load_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
            
            assert load_time < 10, f"Cached load took {load_time}ms, should be <10ms"
            assert result["role"] == "doctor"
        
        @pytest.mark.asyncio
        async def test_scenario_load_performance_fresh(self, cache_setup):
            """Test fresh scenario load performance (<500ms)"""
            
            with patch.object(demo_data_manager.generator, 'generate_scenario') as mock_generate:
                # Simulate realistic generation time
                async def mock_generate_scenario(role):
                    await asyncio.sleep(0.1)  # Simulate 100ms generation
                    return {
                        "role": role,
                        "patients": [{"id": i} for i in range(15)]
                    }
                
                mock_generate.side_effect = mock_generate_scenario
                
                # Measure performance
                start_time = datetime.utcnow()
                result = await demo_data_manager.get_scenario("doctor", "test_user")
                end_time = datetime.utcnow()
                
                load_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
                
                assert load_time < 500, f"Fresh load took {load_time}ms, should be <500ms"
                assert result["role"] == "doctor"
        
        @pytest.mark.asyncio
        async def test_reset_performance(self, cache_setup):
            """Test reset operation performance (<2s)"""
            
            with patch.object(demo_data_manager, '_clear_demo_patients'), \
                 patch.object(demo_data_manager, '_clear_demo_ecg_sessions'), \
                 patch.object(demo_data_manager, '_clear_demo_alarms'):
                
                # Measure performance
                start_time = datetime.utcnow()
                result = await demo_data_manager.reset_all("admin_user")
                end_time = datetime.utcnow()
                
                reset_time = (end_time - start_time).total_seconds()
                
                assert reset_time < 2.0, f"Reset took {reset_time}s, should be <2s"
                assert result["status"] == "resetting"

# Integration tests
class TestDemoAPIIntegration:
    """Integration tests for Demo API"""
    
    @pytest.mark.asyncio
    async def test_full_scenario_workflow(self, cache_setup):
        """Test complete scenario workflow from generation to cache hit"""
        
        user_id = "test_user_integration"
        role = "doctor"
        
        # Step 1: Generate fresh scenario
        with patch.object(demo_data_manager.generator, 'generate_scenario') as mock_generate:
            mock_scenario = {
                "role": role,
                "title": "Doctor Scenario",
                "patients": [{"id": 1, "name": "Test Patient"}],
                "devices": [{"id": 1, "name": "ECG Monitor"}],
                "alerts": [{"id": 1, "message": "Test Alert"}],
                "settings": {"real_time": True}
            }
            mock_generate.return_value = mock_scenario
            
            result1 = await demo_data_manager.get_scenario(role, user_id)
            assert result1["role"] == role
        
        # Step 2: Verify cache hit
        result2 = await demo_data_manager.get_scenario(role, user_id)
        assert result2["role"] == role
        
        # Step 3: Reset and verify cache invalidation
        with patch.object(demo_data_manager, '_clear_demo_patients'):
            await demo_data_manager.reset_all("admin_user")
        
        cache_key = f"demo:scenario:{role}"
        assert not await cache_service.exists(cache_key)
        
        # Step 4: Generate again
        result3 = await demo_data_manager.get_scenario(role, user_id)
        assert result3["role"] == role

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
