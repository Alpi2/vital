package com.vitalstream.hl7.service;

import ca.uhn.hl7v2.DefaultHapiContext;
import ca.uhn.hl7v2.HL7Exception;
import ca.uhn.hl7v2.HapiContext;
import ca.uhn.hl7v2.model.Message;
import ca.uhn.hl7v2.model.v25.message.ORU_R01;
import ca.uhn.hl7v2.model.v25.segment.*;
import ca.uhn.hl7v2.model.v25.group.ORU_R01_PATIENT_RESULT;
import ca.uhn.hl7v2.model.v25.group.ORU_R01_PATIENT;
import ca.uhn.hl7v2.model.v25.group.ORU_R01_ORDER_OBSERVATION;
import ca.uhn.hl7v2.model.v25.group.ORU_R01_OBSERVATION;
import ca.uhn.hl7v2.parser.Parser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * HL7 v2.x Message Processing Service
 * 
 * Handles HL7 message parsing, creation, and validation
 * Compliant with HL7 v2.5 standard
 */
@Service
public class HL7Service {

    private static final Logger log = LoggerFactory.getLogger(HL7Service.class);
    private final HapiContext context;
    private final Parser parser;
    private static final DateTimeFormatter HL7_DATE_FORMAT = 
        DateTimeFormatter.ofPattern("yyyyMMddHHmmss");

    public HL7Service() {
        this.context = new DefaultHapiContext();
        this.parser = context.getPipeParser();
    }

    /**
     * Parse HL7 message from string
     */
    public Message parseMessage(String messageString) throws HL7Exception {
        log.info("Parsing HL7 message");
        return parser.parse(messageString);
    }

    /**
     * Create ORU^R01 message for vital signs observation
     */
    public ORU_R01 createObservationMessage(
            String patientId,
            String observationType,
            String observationValue,
            String unit) throws HL7Exception, IOException {
        
        log.info("Creating ORU^R01 message for patient: {}", patientId);
        
        ORU_R01 message = new ORU_R01();
        message.initQuickstart("ORU", "R01", "P");
        
        // MSH - Message Header
        MSH msh = message.getMSH();
        msh.getSendingApplication().getNamespaceID().setValue("VitalStream");
        msh.getSendingFacility().getNamespaceID().setValue("Hospital");
        msh.getReceivingApplication().getNamespaceID().setValue("HIS");
        msh.getReceivingFacility().getNamespaceID().setValue("Hospital");
        msh.getDateTimeOfMessage().getTime().setValue(
            LocalDateTime.now().format(HL7_DATE_FORMAT));
        msh.getMessageControlID().setValue(UUID.randomUUID().toString());
        msh.getVersionID().getVersionID().setValue("2.5");
        
        // PID - Patient Identification
        ORU_R01_PATIENT_RESULT patientResult = message.getPATIENT_RESULT();
        ORU_R01_PATIENT patient = patientResult.getPATIENT();
        PID pid = patient.getPID();
        pid.getPatientID().getIDNumber().setValue(patientId);
        
        // OBR - Observation Request
        ORU_R01_ORDER_OBSERVATION order = patientResult.getORDER_OBSERVATION();
        OBR obr = order.getOBR();
        obr.getSetIDOBR().setValue("1");
        obr.getUniversalServiceIdentifier().getIdentifier().setValue("VITALS");
        obr.getUniversalServiceIdentifier().getText().setValue("Vital Signs");
        obr.getObservationDateTime().getTime().setValue(
            LocalDateTime.now().format(HL7_DATE_FORMAT));
        
        // OBX - Observation Result
        ORU_R01_OBSERVATION observation = order.getOBSERVATION();
        OBX obx = observation.getOBX();
        obx.getSetIDOBX().setValue("1");
        obx.getValueType().setValue("NM"); // Numeric
        obx.getObservationIdentifier().getIdentifier().setValue(observationType);
        obx.getObservationIdentifier().getText().setValue(observationType);
        obx.getObservationValue(0).getData().parse(observationValue);
        obx.getUnits().getIdentifier().setValue(unit);
        obx.getObservationResultStatus().setValue("F"); // Final
        
        log.info("ORU^R01 message created successfully");
        return message;
    }

    /**
     * Encode message to string
     */
    public String encodeMessage(Message message) throws HL7Exception {
        return parser.encode(message);
    }

    /**
     * Validate HL7 message
     */
    public boolean validateMessage(String messageString) {
        try {
            Message message = parseMessage(messageString);
            return message != null;
        } catch (HL7Exception e) {
            log.error("Message validation failed", e);
            return false;
        }
    }

    /**
     * Extract patient ID from message
     */
    public String extractPatientId(Message message) throws HL7Exception {
        if (message instanceof ORU_R01) {
            ORU_R01 oru = (ORU_R01) message;
            return oru.getPATIENT_RESULT().getPATIENT().getPID()
                .getPatientID().getIDNumber().getValue();
        }
        throw new HL7Exception("Unsupported message type");
    }
}
