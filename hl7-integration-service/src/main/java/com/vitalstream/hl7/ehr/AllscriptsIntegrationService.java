package com.vitalstream.hl7.ehr;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import org.springframework.core.ParameterizedTypeReference;

import java.util.*;

/**
 * Allscripts EHR Integration Service
 * 
 * Integrates with Allscripts using:
 * - Allscripts Unity API
 * - HL7 v2.x messaging
 * - SOAP web services
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class AllscriptsIntegrationService {

    private static final Logger log = LoggerFactory.getLogger(AllscriptsIntegrationService.class);

    @Value("${integration.ehr.allscripts.base-url}")
    private String allscriptsBaseUrl;

    @Value("${integration.ehr.allscripts.app-name}")
    private String appName;

    @Value("${integration.ehr.allscripts.username}")
    private String username;

    @Value("${integration.ehr.allscripts.password}")
    private String password;

    private final RestTemplate restTemplate;
    private String sessionToken;

    public AllscriptsIntegrationService() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Authenticate with Allscripts Unity API
     */
    public String authenticate() {
        if (sessionToken != null) {
            return sessionToken;
        }

        try {
            String url = allscriptsBaseUrl + "/Unity/UnityService.svc/json/GetToken";
            
            Map<String, Object> request = new HashMap<>();
            request.put("Username", username);
            request.put("Password", password);
            request.put("Appname", appName);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(url, HttpMethod.POST, entity, new ParameterizedTypeReference<Map<String, Object>>() {});
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                sessionToken = (String) response.getBody().get("Token");
                log.info("Allscripts authentication successful");
                return sessionToken;
            }
            
        } catch (Exception e) {
            log.error("Allscripts authentication failed", e);
        }
        
        return null;
    }

    /**
     * Get patient demographics
     */
    public Map<String, Object> getPatient(String patientId) {
        String token = authenticate();
        if (token == null) {
            return null;
        }

        try {
            String url = allscriptsBaseUrl + "/Unity/UnityService.svc/json/GetPatient";
            
            Map<String, Object> request = new HashMap<>();
            request.put("Token", token);
            request.put("PatientID", patientId);
            request.put("Appname", appName);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(url, HttpMethod.POST, entity, new ParameterizedTypeReference<Map<String, Object>>() {});
            
            if (response.getStatusCode() == HttpStatus.OK) {
                return response.getBody();
            }
            
        } catch (Exception e) {
            log.error("Error getting patient from Allscripts", e);
        }
        
        return null;
    }

    /**
     * Save vital signs to Allscripts
     */
    public boolean saveVitalSigns(String patientId, Map<String, Object> vitalSigns) {
        String token = authenticate();
        if (token == null) {
            return false;
        }

        try {
            String url = allscriptsBaseUrl + "/Unity/UnityService.svc/json/SaveVitalSigns";
            
            Map<String, Object> request = new HashMap<>();
            request.put("Token", token);
            request.put("PatientID", patientId);
            request.put("VitalSigns", vitalSigns);
            request.put("Appname", appName);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(url, HttpMethod.POST, entity, new ParameterizedTypeReference<Map<String, Object>>() {});
            
            return response.getStatusCode() == HttpStatus.OK;
            
        } catch (Exception e) {
            log.error("Error saving vital signs to Allscripts", e);
        }
        
        return false;
    }
}
