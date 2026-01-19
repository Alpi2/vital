package com.vitalstream.hl7.lis;

import ca.uhn.hl7v2.model.v25.message.ORU_R01;
import ca.uhn.hl7v2.model.v25.segment.OBX;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Laboratory Information System (LIS) Integration Service
 * 
 * Handles integration with laboratory systems for:
 * - Lab result reception (HL7 ORU^R01)
 * - Critical value alerts
 * - Auto-ordering (HL7 ORM^O01)
 * - Result display and trending
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class LISIntegrationService {

    private static final Logger log = LoggerFactory.getLogger(LISIntegrationService.class);

    /**
     * Lab result data
     */
    public static class LabResult {
        private String patientId;
        private String testCode;
        private String testName;
        private String value;
        private String unit;
        private String referenceRange;
        private String abnormalFlag;  // N=Normal, H=High, L=Low, HH=Critical High, LL=Critical Low
        private Date observationTime;
        private String status;  // F=Final, P=Preliminary, C=Corrected
        
        // Getters and setters
        public String getPatientId() { return patientId; }
        public void setPatientId(String patientId) { this.patientId = patientId; }
        
        public String getTestCode() { return testCode; }
        public void setTestCode(String testCode) { this.testCode = testCode; }
        
        public String getTestName() { return testName; }
        public void setTestName(String testName) { this.testName = testName; }
        
        public String getValue() { return value; }
        public void setValue(String value) { this.value = value; }
        
        public String getUnit() { return unit; }
        public void setUnit(String unit) { this.unit = unit; }
        
        public String getReferenceRange() { return referenceRange; }
        public void setReferenceRange(String referenceRange) { this.referenceRange = referenceRange; }
        
        public String getAbnormalFlag() { return abnormalFlag; }
        public void setAbnormalFlag(String abnormalFlag) { this.abnormalFlag = abnormalFlag; }
        
        public Date getObservationTime() { return observationTime; }
        public void setObservationTime(Date observationTime) { this.observationTime = observationTime; }
        
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public boolean isCritical() {
            return "HH".equals(abnormalFlag) || "LL".equals(abnormalFlag);
        }
    }

    /**
     * Parse HL7 ORU^R01 message (lab results)
     * 
     * @param message HL7 ORU^R01 message
     * @return List of lab results
     */
    public List<LabResult> parseLabResults(ORU_R01 message) {
        List<LabResult> results = new ArrayList<>();
        
        try {
            String patientId = message.getPATIENT_RESULT().getPATIENT().getPID().getPatientID().getIDNumber().getValue();
            
            // Iterate through observation groups
            int numOBR = message.getPATIENT_RESULT().getORDER_OBSERVATIONReps();
            for (int i = 0; i < numOBR; i++) {
                var orderObs = message.getPATIENT_RESULT().getORDER_OBSERVATION(i);
                
                // Iterate through observations (OBX segments)
                int numOBX = orderObs.getOBSERVATIONReps();
                for (int j = 0; j < numOBX; j++) {
                    OBX obx = orderObs.getOBSERVATION(j).getOBX();
                    
                    LabResult result = new LabResult();
                    result.setPatientId(patientId);
                    result.setTestCode(obx.getObservationIdentifier().getIdentifier().getValue());
                    result.setTestName(obx.getObservationIdentifier().getText().getValue());
                    result.setValue(obx.getObservationValue(0).getData().toString());
                    result.setUnit(obx.getUnits().getIdentifier().getValue());
                    result.setReferenceRange(obx.getReferencesRange().getValue());
                    result.setAbnormalFlag(obx.getAbnormalFlags(0).getValue());
                    result.setStatus(obx.getObservationResultStatus().getValue());
                    
                    // Parse observation time
                    String obsTime = obx.getDateTimeOfTheObservation().getTime().getValue();
                    if (obsTime != null && !obsTime.isEmpty()) {
                        result.setObservationTime(parseHL7DateTime(obsTime));
                    }
                    
                    results.add(result);
                    
                    log.debug("Parsed lab result: {} = {} {} ({})",
                        result.getTestName(), result.getValue(), result.getUnit(), result.getAbnormalFlag());
                }
            }
            
        } catch (Exception e) {
            log.error("Error parsing lab results", e);
        }
        
        return results;
    }

    /**
     * Get troponin results
     * 
     * @param results All lab results
     * @return Troponin results
     */
    public List<LabResult> getTroponinResults(List<LabResult> results) {
        return results.stream()
            .filter(r -> r.getTestCode().matches("(?i).*TROP.*|.*TNI.*|.*TNT.*"))
            .toList();
    }

    /**
     * Get electrolyte results (Na, K, Cl, CO2)
     * 
     * @param results All lab results
     * @return Electrolyte results
     */
    public Map<String, LabResult> getElectrolyteResults(List<LabResult> results) {
        Map<String, LabResult> electrolytes = new HashMap<>();
        
        for (LabResult result : results) {
            String code = result.getTestCode().toUpperCase();
            if (code.contains("NA") || code.contains("SODIUM")) {
                electrolytes.put("Sodium", result);
            } else if (code.contains("K") || code.contains("POTASSIUM")) {
                electrolytes.put("Potassium", result);
            } else if (code.contains("CL") || code.contains("CHLORIDE")) {
                electrolytes.put("Chloride", result);
            } else if (code.contains("CO2") || code.contains("BICARB")) {
                electrolytes.put("CO2", result);
            }
        }
        
        return electrolytes;
    }

    /**
     * Get blood gas results (pH, pO2, pCO2, HCO3, BE)
     * 
     * @param results All lab results
     * @return Blood gas results
     */
    public Map<String, LabResult> getBloodGasResults(List<LabResult> results) {
        Map<String, LabResult> bloodGas = new HashMap<>();
        
        for (LabResult result : results) {
            String code = result.getTestCode().toUpperCase();
            if (code.contains("PH")) {
                bloodGas.put("pH", result);
            } else if (code.contains("PO2") || code.contains("PAO2")) {
                bloodGas.put("pO2", result);
            } else if (code.contains("PCO2") || code.contains("PACO2")) {
                bloodGas.put("pCO2", result);
            } else if (code.contains("HCO3") || code.contains("BICARB")) {
                bloodGas.put("HCO3", result);
            } else if (code.contains("BE") || code.contains("BASE")) {
                bloodGas.put("BaseExcess", result);
            }
        }
        
        return bloodGas;
    }

    /**
     * Check for critical values
     * 
     * @param results Lab results
     * @return List of critical results
     */
    public List<LabResult> getCriticalResults(List<LabResult> results) {
        return results.stream()
            .filter(LabResult::isCritical)
            .toList();
    }

    /**
     * Generate auto-order for lab tests
     * 
     * @param patientId Patient ID
     * @param testCodes List of test codes to order
     * @param priority Priority (STAT, ASAP, ROUTINE)
     * @return HL7 ORM^O01 message
     */
    public String generateLabOrder(String patientId, List<String> testCodes, String priority) {
        // Implementation pending: ORM^O01 message generation
        log.info("Generating lab order for patient {} with {} tests (priority: {})",
            patientId, testCodes.size(), priority);
        
        // This would generate an HL7 ORM^O01 message
        return "ORM^O01 message placeholder";
    }

    /**
     * Parse HL7 date/time
     */
    private Date parseHL7DateTime(String hl7DateTime) {
        try {
            // HL7 format: YYYYMMDDHHMMSS
            java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("yyyyMMddHHmmss");
            return sdf.parse(hl7DateTime);
        } catch (Exception e) {
            log.warn("Failed to parse HL7 date/time: {}", hl7DateTime);
            return new Date();
        }
    }
}
