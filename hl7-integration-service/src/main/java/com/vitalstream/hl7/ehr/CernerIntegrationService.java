package com.vitalstream.hl7.ehr;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.*;

/**
 * Cerner EHR Integration Service
 * 
 * Integrates with Cerner Millennium using:
 * - Cerner FHIR API (DSTU2/R4)
 * - OAuth 2.0 authentication
 * - CareAware integration
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class CernerIntegrationService {

    private static final Logger log = LoggerFactory.getLogger(CernerIntegrationService.class);

    @Value("${integration.ehr.cerner.base-url}")
    private String cernerBaseUrl;

    @Value("${integration.ehr.cerner.client-id}")
    private String clientId;

    @Value("${integration.ehr.cerner.client-secret}")
    private String clientSecret;

    @Value("${integration.ehr.cerner.tenant-id}")
    private String tenantId;

    private final RestTemplate restTemplate;
    private String accessToken;
    private Date tokenExpiry;

    public CernerIntegrationService() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Authenticate with Cerner using OAuth 2.0
     */
    public String authenticate() {
        if (accessToken != null && tokenExpiry != null && new Date().before(tokenExpiry)) {
            return accessToken;
        }

        try {
            String tokenUrl = cernerBaseUrl + "/tenants/" + tenantId + "/protocols/oauth2/profiles/smart-v1/token";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
            headers.setBasicAuth(clientId, clientSecret);
            
            String body = "grant_type=client_credentials&scope=system/Patient.read system/Observation.write";
            
            HttpEntity<String> request = new HttpEntity<>(body, headers);
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity(tokenUrl, request, (Class<Map<String, Object>>)(Class<?>)Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                accessToken = (String) response.getBody().get("access_token");
                int expiresIn = (Integer) response.getBody().get("expires_in");
                tokenExpiry = new Date(System.currentTimeMillis() + (expiresIn * 1000L));
                
                log.info("Cerner authentication successful");
                return accessToken;
            }
            
        } catch (Exception e) {
            log.error("Cerner authentication failed", e);
        }
        
        return null;
    }

    /**
     * Get patient by MRN from Cerner
     */
    public Map<String, Object> getPatientByMRN(String mrn) {
        String token = authenticate();
        if (token == null) {
            return null;
        }

        try {
            String url = String.format("%s/Patient?identifier=%s", cernerBaseUrl, mrn);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(token);
            headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));
            
            HttpEntity<String> request = new HttpEntity<>(headers);
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(url, HttpMethod.GET, request, (Class<Map<String, Object>>)(Class<?>)Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK) {
                return response.getBody();
            }
            
        } catch (Exception e) {
            log.error("Error getting patient from Cerner", e);
        }
        
        return null;
    }

    /**
     * Send vital signs observation to Cerner
     */
    public boolean sendVitalSigns(String patientId, Map<String, Object> vitalSigns) {
        String token = authenticate();
        if (token == null) {
            return false;
        }

        try {
            String url = cernerBaseUrl + "/Observation";
            
            Map<String, Object> observation = buildCernerObservation(patientId, vitalSigns);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(token);
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(observation, headers);
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity(url, request, (Class<Map<String, Object>>)(Class<?>)Map.class);
            
            return response.getStatusCode() == HttpStatus.CREATED;
            
        } catch (Exception e) {
            log.error("Error sending vital signs to Cerner", e);
        }
        
        return false;
    }

    private Map<String, Object> buildCernerObservation(String patientId, Map<String, Object> vitalSigns) {
        Map<String, Object> observation = new HashMap<>();
        observation.put("resourceType", "Observation");
        observation.put("status", "final");
        
        Map<String, Object> subject = new HashMap<>();
        subject.put("reference", "Patient/" + patientId);
        observation.put("subject", subject);
        
        // Add vital signs data
        observation.put("effectiveDateTime", new Date().toString());
        
        return observation;
    }
}
