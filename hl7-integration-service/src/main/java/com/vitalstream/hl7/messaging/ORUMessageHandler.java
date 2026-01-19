package com.vitalstream.hl7.messaging;

import ca.uhn.hl7v2.HL7Exception;
import ca.uhn.hl7v2.model.v25.message.ORU_R01;
import ca.uhn.hl7v2.model.v25.segment.*;
import ca.uhn.hl7v2.model.v25.group.ORU_R01_ORDER_OBSERVATION;
import ca.uhn.hl7v2.model.v25.group.ORU_R01_OBSERVATION;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * ORU (Observation Result) Message Handler
 * 
 * Handles HL7 ORU^R01 messages for:
 * - Laboratory results
 * - Vital signs
 * - Radiology results
 * - Pathology results
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Component
public class ORUMessageHandler {

    private static final Logger log = LoggerFactory.getLogger(ORUMessageHandler.class);

    /**
     * Laboratory result
     */
    public static class LabResult {
        private String testCode;
        private String testName;
        private String value;
        private String unit;
        private String referenceRange;
        private String abnormalFlag;  // N=Normal, H=High, L=Low, HH=Critical High, LL=Critical Low
        private String status;         // F=Final, P=Preliminary, C=Corrected
        private String observationDateTime;
        
        // Getters and setters
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
        
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public String getObservationDateTime() { return observationDateTime; }
        public void setObservationDateTime(String observationDateTime) { 
            this.observationDateTime = observationDateTime; 
        }
        
        public boolean isCritical() {
            return "HH".equals(abnormalFlag) || "LL".equals(abnormalFlag);
        }
    }

    /**
     * Parse ORU^R01 message
     */
    public Map<String, Object> parseORUMessage(ORU_R01 message) throws HL7Exception {
        Map<String, Object> result = new HashMap<>();
        
        // Parse MSH
        MSH msh = message.getMSH();
        result.put("messageId", msh.getMessageControlID().getValue());
        result.put("timestamp", msh.getDateTimeOfMessage().getTime().getValue());
        result.put("sendingApplication", msh.getSendingApplication().getNamespaceID().getValue());
        
        // Parse PID
        PID pid = message.getPATIENT_RESULT().getPATIENT().getPID();
        result.put("patientId", pid.getPatientID().getIDNumber().getValue());
        result.put("mrn", pid.getPatientIdentifierList(0).getIDNumber().getValue());
        
        String lastName = pid.getPatientName(0).getFamilyName().getSurname().getValue();
        String firstName = pid.getPatientName(0).getGivenName().getValue();
        result.put("patientName", firstName + " " + lastName);
        
        // Parse observations
        List<LabResult> labResults = new ArrayList<>();
        
        int orderObsCount = message.getPATIENT_RESULT().getORDER_OBSERVATIONReps();
        for (int i = 0; i < orderObsCount; i++) {
            ORU_R01_ORDER_OBSERVATION orderObs = message.getPATIENT_RESULT().getORDER_OBSERVATION(i);
            
            // Parse OBR (Observation Request)
            OBR obr = orderObs.getOBR();
            String orderCode = obr.getUniversalServiceIdentifier().getIdentifier().getValue();
            String orderName = obr.getUniversalServiceIdentifier().getText().getValue();
            
            result.put("orderCode", orderCode);
            result.put("orderName", orderName);
            result.put("orderDateTime", obr.getObservationDateTime().getTime().getValue());
            
            // Parse OBX (Observation/Result)
            int obsCount = orderObs.getOBSERVATIONReps();
            for (int j = 0; j < obsCount; j++) {
                ORU_R01_OBSERVATION obs = orderObs.getOBSERVATION(j);
                OBX obx = obs.getOBX();
                
                LabResult labResult = new LabResult();
                
                // Test code and name
                labResult.setTestCode(obx.getObservationIdentifier().getIdentifier().getValue());
                labResult.setTestName(obx.getObservationIdentifier().getText().getValue());
                
                // Value
                String valueType = obx.getValueType().getValue();
                if ("NM".equals(valueType)) { // Numeric
                    labResult.setValue(obx.getObservationValue(0).getData().toString());
                } else if ("ST".equals(valueType)) { // String
                    labResult.setValue(obx.getObservationValue(0).getData().toString());
                }
                
                // Unit
                labResult.setUnit(obx.getUnits().getIdentifier().getValue());
                
                // Reference range
                labResult.setReferenceRange(obx.getReferencesRange().getValue());
                
                // Abnormal flag
                if (obx.getAbnormalFlags().length > 0) {
                    labResult.setAbnormalFlag(obx.getAbnormalFlags(0).getValue());
                }
                
                // Status
                labResult.setStatus(obx.getObservationResultStatus().getValue());
                
                // Observation date/time
                labResult.setObservationDateTime(obx.getDateTimeOfTheObservation().getTime().getValue());
                
                labResults.add(labResult);
                
                // Log critical values
                if (labResult.isCritical()) {
                    log.warn("CRITICAL LAB VALUE: {} = {} {} (Patient: {})",
                            labResult.getTestName(),
                            labResult.getValue(),
                            labResult.getUnit(),
                            result.get("mrn"));
                }
            }
        }
        
        result.put("labResults", labResults);
        result.put("resultCount", labResults.size());
        
        log.info("Parsed ORU^R01 message: {} results for patient {}", 
                labResults.size(), result.get("mrn"));
        
        return result;
    }

    /**
     * Extract specific lab value by test code
     */
    public LabResult getLabValue(Map<String, Object> oruData, String testCode) {
        @SuppressWarnings("unchecked")
        List<LabResult> results = (List<LabResult>) oruData.get("labResults");
        
        if (results != null) {
            for (LabResult result : results) {
                if (testCode.equals(result.getTestCode())) {
                    return result;
                }
            }
        }
        
        return null;
    }

    /**
     * Get all critical values
     */
    public List<LabResult> getCriticalValues(Map<String, Object> oruData) {
        List<LabResult> criticalValues = new ArrayList<>();
        
        @SuppressWarnings("unchecked")
        List<LabResult> results = (List<LabResult>) oruData.get("labResults");
        
        if (results != null) {
            for (LabResult result : results) {
                if (result.isCritical()) {
                    criticalValues.add(result);
                }
            }
        }
        
        return criticalValues;
    }
}
