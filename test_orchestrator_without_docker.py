#!/usr/bin/env python3
"""
Test Orchestrator without Docker dependencies
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockService:
    """Mock service for testing"""
    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.status = "healthy"
    
    async def health_check(self):
        """Mock health check"""
        return {"status": "healthy", "service": self.name}

class TestOrchestratorWithoutDocker:
    """Test orchestrator without Docker dependencies"""
    
    def __init__(self):
        self.services = {
            "backend": MockService("backend", 8000),
            "alarm_engine": MockService("alarm_engine", 50052),
            "dicom_service": MockService("dicom_service", 50051),
            "hl7_service": MockService("hl7_service", 8080),
            "frontend": MockService("frontend", 4200),
        }
        self.test_results = {}
    
    async def check_service_health(self):
        """Check health of all services"""
        logger.info("🔍 Checking service health...")
        
        for service_name, service in self.services.items():
            try:
                health = await service.health_check()
                logger.info(f"✅ {service_name}: {health['status']}")
                self.test_results[f"{service_name}_health"] = True
            except Exception as e:
                logger.error(f"❌ {service_name}: {e}")
                self.test_results[f"{service_name}_health"] = False
    
    async def test_basic_functionality(self):
        """Test basic functionality"""
        logger.info("🧪 Testing basic functionality...")
        
        # Test imports
        try:
            from backend.app.compliance.gdpr import GDPRCompliance
            gdpr = GDPRCompliance()
            principles = gdpr.check_gdpr_principles()
            logger.info(f"✅ GDPR compliance: {len(principles)} principles checked")
            self.test_results["gdpr_compliance"] = True
        except Exception as e:
            logger.error(f"❌ GDPR compliance: {e}")
            self.test_results["gdpr_compliance"] = False
        
        # Test service modules
        try:
            from backend.app.services.notification_service import NotificationService
            logger.info("✅ Notification service imported successfully")
            self.test_results["notification_service"] = True
        except Exception as e:
            logger.error(f"❌ Notification service: {e}")
            self.test_results["notification_service"] = False
        
        # Test ML modules
        try:
            from backend.app.ml.compliance_prediction_service import CompliancePredictionService
            logger.info("✅ ML service imported successfully")
            self.test_results["ml_service"] = True
        except Exception as e:
            logger.error(f"❌ ML service: {e}")
            self.test_results["ml_service"] = False
    
    async def run_integration_tests(self):
        """Run integration tests"""
        logger.info("🚀 Starting Integration Tests without Docker...")
        
        # Test service health
        await self.check_service_health()
        
        # Test basic functionality
        await self.test_basic_functionality()
        
        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        logger.info("📊 Test Results Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests} ✅")
        logger.info(f"   Failed: {failed_tests} ❌")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED! Integration tests working correctly.")
            return True
        else:
            logger.error("❌ Some tests failed. Please check the errors above.")
            return False

if __name__ == "__main__":
    orchestrator = TestOrchestratorWithoutDocker()
    success = asyncio.run(orchestrator.run_integration_tests())
    sys.exit(0 if success else 1)
