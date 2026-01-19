package com.vitalstream.hl7.messaging;

import ca.uhn.hl7v2.HL7Exception;
import ca.uhn.hl7v2.model.v25.message.ADT_A01;
import ca.uhn.hl7v2.model.v25.segment.*;
import ca.uhn.hl7v2.model.v25.datatype.XAD;
import ca.uhn.hl7v2.model.v25.datatype.PL;
import ca.uhn.hl7v2.model.v25.datatype.XCN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * ADT (Admission/Discharge/Transfer) Message Handler
 * 
 * Handles HL7 ADT messages:
 * - A01: Patient Admission
 * - A02: Patient Transfer
 * - A03: Patient Discharge
 * - A04: Patient Registration
 * - A08: Patient Update
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Component
public class ADTMessageHandler {

    private static final Logger log = LoggerFactory.getLogger(ADTMessageHandler.class);

    /**
     * Patient admission event (A01)
     */
    public Map<String, Object> handleAdmission(ADT_A01 message) throws HL7Exception {
        Map<String, Object> patientData = new HashMap<>();
        
        // Parse MSH (Message Header)
        MSH msh = message.getMSH();
        patientData.put("messageId", msh.getMessageControlID().getValue());
        patientData.put("timestamp", msh.getDateTimeOfMessage().getTime().getValue());
        
        // Parse EVN (Event Type)
        EVN evn = message.getEVN();
        patientData.put("eventType", evn.getEventTypeCode().getValue());
        patientData.put("eventTimestamp", evn.getRecordedDateTime().getTime().getValue());
        
        // Parse PID (Patient Identification)
        PID pid = message.getPID();
        patientData.put("patientId", pid.getPatientID().getIDNumber().getValue());
        patientData.put("mrn", pid.getPatientIdentifierList(0).getIDNumber().getValue());
        
        // Patient name
        String lastName = pid.getPatientName(0).getFamilyName().getSurname().getValue();
        String firstName = pid.getPatientName(0).getGivenName().getValue();
        patientData.put("name", firstName + " " + lastName);
        
        // Demographics
        patientData.put("dateOfBirth", pid.getDateTimeOfBirth().getTime().getValue());
        patientData.put("gender", pid.getAdministrativeSex().getValue());
        
        // Address
        if (pid.getPatientAddress().length > 0) {
            XAD address = pid.getPatientAddress(0);
            patientData.put("address", address.getStreetAddress().getStreetOrMailingAddress().getValue());
            patientData.put("city", address.getCity().getValue());
            patientData.put("state", address.getStateOrProvince().getValue());
            patientData.put("zip", address.getZipOrPostalCode().getValue());
        }
        
        // Phone
        if (pid.getPhoneNumberHome().length > 0) {
            patientData.put("phone", pid.getPhoneNumberHome(0).getTelephoneNumber().getValue());
        }
        
        // Parse PV1 (Patient Visit)
        PV1 pv1 = message.getPV1();
        patientData.put("patientClass", pv1.getPatientClass().getValue()); // I=Inpatient, O=Outpatient, E=Emergency
        
        // Location
        if (pv1.getAssignedPatientLocation() != null) {
            PL location = pv1.getAssignedPatientLocation();
            patientData.put("facility", location.getFacility().getNamespaceID().getValue());
            patientData.put("building", location.getBuilding().getValue());
            patientData.put("floor", location.getFloor().getValue());
            patientData.put("room", location.getRoom().getValue());
            patientData.put("bed", location.getBed().getValue());
        }
        
        // Attending doctor
        if (pv1.getAttendingDoctor().length > 0) {
            XCN doctor = pv1.getAttendingDoctor(0);
            String doctorName = doctor.getGivenName().getValue() + " " + 
                               doctor.getFamilyName().getSurname().getValue();
            patientData.put("attendingDoctor", doctorName);
            patientData.put("doctorId", doctor.getIDNumber().getValue());
        }
        
        // Admission date/time
        patientData.put("admissionDateTime", pv1.getAdmitDateTime().getTime().getValue());
        
        log.info("Processed ADT^A01 admission for patient: {}", patientData.get("mrn"));
        
        return patientData;
    }

    /**
     * Patient transfer event (A02)
     */
    public Map<String, Object> handleTransfer(ADT_A01 message) throws HL7Exception {
        Map<String, Object> transferData = handleAdmission(message); // Same structure
        
        PV1 pv1 = message.getPV1();
        
        // Prior location
        if (pv1.getPriorPatientLocation() != null) {
            PL priorLocation = pv1.getPriorPatientLocation();
            transferData.put("priorRoom", priorLocation.getRoom().getValue());
            transferData.put("priorBed", priorLocation.getBed().getValue());
        }
        
        log.info("Processed ADT^A02 transfer for patient: {}", transferData.get("mrn"));
        
        return transferData;
    }

    /**
     * Patient discharge event (A03)
     */
    public Map<String, Object> handleDischarge(ADT_A01 message) throws HL7Exception {
        Map<String, Object> dischargeData = handleAdmission(message);
        
        PV1 pv1 = message.getPV1();
        dischargeData.put("dischargeDateTime", pv1.getDischargeDateTime(0).getTime().getValue());
        dischargeData.put("dischargeDisposition", pv1.getDischargeDisposition().getValue());
        
        log.info("Processed ADT^A03 discharge for patient: {}", dischargeData.get("mrn"));
        
        return dischargeData;
    }

    /**
     * Patient update event (A08)
     */
    public Map<String, Object> handleUpdate(ADT_A01 message) throws HL7Exception {
        Map<String, Object> updateData = handleAdmission(message);
        
        log.info("Processed ADT^A08 update for patient: {}", updateData.get("mrn"));
        
        return updateData;
    }
}
