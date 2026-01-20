import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestVitalStreamE2E:
    @pytest.fixture
    def driver(self):
        driver = webdriver.Chrome()
        driver.implicitly_wait(10)
        yield driver
        driver.quit()

    def test_login_flow(self, driver):
        """Test complete login flow"""
        driver.get("http://localhost:4200/login")
        
        # Enter credentials
        username = driver.find_element(By.ID, "username")
        password = driver.find_element(By.ID, "password")
        
        username.send_keys("test@vitalstream.com")
        password.send_keys("testpassword")
        
        # Click login
        login_btn = driver.find_element(By.ID, "login-btn")
        login_btn.click()
        
        # Wait for dashboard
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "dashboard"))
        )
        
        assert "dashboard" in driver.current_url

    def test_patient_list_view(self, driver):
        """Test patient list viewing"""
        # Login first
        self.test_login_flow(driver)
        
        # Navigate to patients
        patients_link = driver.find_element(By.ID, "patients-link")
        patients_link.click()
        
        # Wait for patient list
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "patient-card"))
        )
        
        patient_cards = driver.find_elements(By.CLASS_NAME, "patient-card")
        assert len(patient_cards) > 0

    def test_alarm_acknowledgment(self, driver):
        """Test alarm acknowledgment flow"""
        self.test_login_flow(driver)
        
        # Navigate to alarms
        alarms_link = driver.find_element(By.ID, "alarms-link")
        alarms_link.click()
        
        # Find first alarm
        alarm = driver.find_element(By.CLASS_NAME, "alarm-item")
        
        # Click acknowledge
        ack_btn = alarm.find_element(By.CLASS_NAME, "ack-btn")
        ack_btn.click()
        
        # Verify acknowledged
        time.sleep(1)
        assert "acknowledged" in alarm.get_attribute("class")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
