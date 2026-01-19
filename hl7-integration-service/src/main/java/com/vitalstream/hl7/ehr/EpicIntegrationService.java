package com.vitalstream.hl7.ehr;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.*;

/**
 * Epic EHR Integration Service
 * 
 * Integrates with Epic Systems using:
 * - Epic Interconnect (FHIR R4)
 * - Epic Web Services (SOAP)
 * - OAuth 2.0 authentication
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class EpicIntegrationService {

    private static final Logger log = LoggerFactory.getLogger(EpicIntegrationService.class);

    @Value("${integration.ehr.epic.base-url}")
    private String epicBaseUrl;

    @Value("${integration.ehr.epic.client-id}")
    private String clientId;

    @Value("${integration.ehr.epic.client-secret}")
    private String clientSecret;

    private final RestTemplate restTemplate;
    private String accessToken;
    private Date tokenExpiry;

    public EpicIntegrationService() {
        this.restTemplate = new RestTemplate();
    }

    /**
     * Patient demographics from Epic
     */
    public static class EpicPatient {
        private String id;              // Epic patient ID (FHIR ID)
        private String mrn;             // Medical Record Number
        private String name;
        private String dateOfBirth;
        private String gender;
        private String address;
        private String phone;
        private String email;
        
        // Getters and setters
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        
        public String getMrn() { return mrn; }
        public void setMrn(String mrn) { this.mrn = mrn; }
        
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public String getDateOfBirth() { return dateOfBirth; }
        public void setDateOfBirth(String dateOfBirth) { this.dateOfBirth = dateOfBirth; }
        
        public String getGender() { return gender; }
        public void setGender(String gender) { this.gender = gender; }
        
        public String getAddress() { return address; }
        public void setAddress(String address) { this.address = address; }
        
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
    }

    /**
     * Authenticate with Epic using OAuth 2.0
     * 
     * @return Access token
     */
    public String authenticate() {
        if (accessToken != null && tokenExpiry != null && new Date().before(tokenExpiry)) {
            return accessToken;
        }

        try {
            String tokenUrl = epicBaseUrl + "/oauth2/token";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
            
            String body = String.format(
                "grant_type=client_credentials&client_id=%s&client_secret=%s",
                clientId, clientSecret
            );
            
            HttpEntity<String> request = new HttpEntity<>(body, headers);
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity(tokenUrl, request, (Class<Map<String, Object>>)(Class<?>)Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                accessToken = (String) response.getBody().get("access_token");
                int expiresIn = (Integer) response.getBody().get("expires_in");
                tokenExpiry = new Date(System.currentTimeMillis() + (expiresIn * 1000L));
                
                log.info("Epic authentication successful, token expires in {} seconds", expiresIn);
                return accessToken;
            }
            
        } catch (Exception e) {
            log.error("Epic authentication failed", e);
        }
        
        return null;
    }

    /**
     * Get patient by MRN
     * 
     * @param mrn Medical Record Number
     * @return Patient information
     */
    public EpicPatient getPatientByMRN(String mrn) {
        String token = authenticate();
        if (token == null) {
            log.error("Cannot get patient: authentication failed");
            return null;
        }

        try {
            String url = String.format("%s/api/FHIR/R4/Patient?identifier=%s", epicBaseUrl, mrn);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(token);
            headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));
            
            HttpEntity<String> request = new HttpEntity<>(headers);
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(url, HttpMethod.GET, request, (Class<Map<String, Object>>)(Class<?>)Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                // Parse FHIR Bundle
                Map<String, Object> bundle = response.getBody();
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> entries = (List<Map<String, Object>>) bundle.get("entry");
                
                if (entries != null && !entries.isEmpty()) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> patientResource = (Map<String, Object>) entries.get(0).get("resource");
                    return parseFHIRPatient(patientResource);
                }
            }
            
        } catch (Exception e) {
            log.error("Error getting patient from Epic", e);
        }
        
        return null;
    }

    /**
     * Send observation to Epic (vital signs)
     * 
     * @param patientId Epic patient ID
     * @param observationType Type (heart-rate, blood-pressure, etc.)
     * @param value Observation value
     * @param unit Unit of measurement
     * @return true if successful
     */
    public boolean sendObservation(String patientId, String observationType, 
                                   String value, String unit) {
        String token = authenticate();
        if (token == null) {
            return false;
        }

        try {
            String url = epicBaseUrl + "/api/FHIR/R4/Observation";
            
            // Build FHIR Observation resource
            Map<String, Object> observation = new HashMap<>();
            observation.put("resourceType", "Observation");
            observation.put("status", "final");
            
            Map<String, Object> code = new HashMap<>();
            code.put("text", observationType);
            observation.put("code", code);
            
            Map<String, Object> subject = new HashMap<>();
            subject.put("reference", "Patient/" + patientId);
            observation.put("subject", subject);
            
            Map<String, Object> valueQuantity = new HashMap<>();
            valueQuantity.put("value", Double.parseDouble(value));
            valueQuantity.put("unit", unit);
            observation.put("valueQuantity", valueQuantity);
            
            observation.put("effectiveDateTime", new Date().toString());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(token);
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(observation, headers);
            @SuppressWarnings("unchecked")
            ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity(url, request, (Class<Map<String, Object>>)(Class<?>)Map.class);
            
            if (response.getStatusCode() == HttpStatus.CREATED) {
                log.info("Observation sent to Epic successfully");
                return true;
            }
            
        } catch (Exception e) {
            log.error("Error sending observation to Epic", e);
        }
        
        return false;
    }

    /**
     * Parse FHIR Patient resource
     */
    private EpicPatient parseFHIRPatient(Map<String, Object> resource) {
        EpicPatient patient = new EpicPatient();
        
        patient.setId((String) resource.get("id"));
        
        // Parse name
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> names = (List<Map<String, Object>>) resource.get("name");
        if (names != null && !names.isEmpty()) {
            Map<String, Object> name = names.get(0);
            @SuppressWarnings("unchecked")
            String given = String.join(" ", (List<String>) name.get("given"));
            String family = (String) name.get("family");
            patient.setName(given + " " + family);
        }
        
        patient.setDateOfBirth((String) resource.get("birthDate"));
        patient.setGender((String) resource.get("gender"));
        
        // Parse identifiers for MRN
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> identifiers = (List<Map<String, Object>>) resource.get("identifier");
        if (identifiers != null) {
            for (Map<String, Object> id : identifiers) {
                @SuppressWarnings("unchecked")
                Map<String, Object> type = (Map<String, Object>) id.get("type");
                if (type != null) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> coding = (List<Map<String, Object>>) type.get("coding");
                    if (coding != null && !coding.isEmpty()) {
                        String code = (String) coding.get(0).get("code");
                        if ("MR".equals(code)) {
                            patient.setMrn((String) id.get("value"));
                            break;
                        }
                    }
                }
            }
        }
        
        return patient;
    }
}
