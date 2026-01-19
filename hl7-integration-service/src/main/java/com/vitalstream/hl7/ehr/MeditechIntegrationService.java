package com.vitalstream.hl7.ehr;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.*;
import java.net.Socket;
import java.util.*;

/**
 * Meditech EHR Integration Service
 * 
 * Integrates with Meditech using:
 * - HL7 v2.x over MLLP (Minimal Lower Layer Protocol)
 * - TCP/IP socket communication
 * - ADT, ORU, ORM message types
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class MeditechIntegrationService {

    private static final Logger log = LoggerFactory.getLogger(MeditechIntegrationService.class);

    @Value("${integration.ehr.meditech.host}")
    private String meditechHost;

    @Value("${integration.ehr.meditech.port}")
    private int meditechPort;

    private static final char START_OF_BLOCK = 0x0B;
    private static final char END_OF_BLOCK = 0x1C;
    private static final char CARRIAGE_RETURN = 0x0D;

    /**
     * Send HL7 message to Meditech via MLLP
     * 
     * @param hl7Message HL7 v2.x message
     * @return ACK message from Meditech
     */
    public String sendHL7Message(String hl7Message) {
        try (Socket socket = new Socket(meditechHost, meditechPort);
             OutputStream out = socket.getOutputStream();
             InputStream in = socket.getInputStream()) {
            
            // Wrap message in MLLP envelope
            String mllpMessage = START_OF_BLOCK + hl7Message + END_OF_BLOCK + CARRIAGE_RETURN;
            
            // Send message
            out.write(mllpMessage.getBytes());
            out.flush();
            
            log.info("Sent HL7 message to Meditech: {} bytes", mllpMessage.length());
            
            // Read ACK response
            ByteArrayOutputStream response = new ByteArrayOutputStream();
            int b;
            while ((b = in.read()) != -1) {
                if (b == END_OF_BLOCK) {
                    break;
                }
                if (b != START_OF_BLOCK) {
                    response.write(b);
                }
            }
            
            String ack = response.toString();
            log.info("Received ACK from Meditech: {}", ack);
            
            return ack;
            
        } catch (Exception e) {
            log.error("Error sending HL7 message to Meditech", e);
        }
        
        return null;
    }

    /**
     * Send ADT^A01 (Patient Admission) message
     */
    public boolean sendAdmission(String patientId, String patientName, String location) {
        String hl7 = buildADTA01Message(patientId, patientName, location);
        String ack = sendHL7Message(hl7);
        return ack != null && ack.contains("AA"); // Application Accept
    }

    /**
     * Send ORU^R01 (Observation Result) message for vital signs
     */
    public boolean sendVitalSigns(String patientId, Map<String, Object> vitalSigns) {
        String hl7 = buildORUR01Message(patientId, vitalSigns);
        String ack = sendHL7Message(hl7);
        return ack != null && ack.contains("AA");
    }

    /**
     * Build ADT^A01 message
     */
    private String buildADTA01Message(String patientId, String patientName, String location) {
        StringBuilder hl7 = new StringBuilder();
        String timestamp = getCurrentTimestamp();
        
        // MSH segment
        hl7.append("MSH|^~\\&|VitalStream|Hospital|Meditech|Hospital|")
           .append(timestamp)
           .append("||");
        hl7.append("ADT^A01|MSG").append(System.currentTimeMillis()).append("|P|2.5\r");
        
        // EVN segment
        hl7.append("EVN|A01|").append(timestamp).append("\r");
        
        // PID segment
        hl7.append("PID|1||").append(patientId).append("|||");
        hl7.append(patientName).append("||19800101|M\r");
        
        // PV1 segment
        hl7.append("PV1|1|I|").append(location).append("\r");
        
        return hl7.toString();
    }

    /**
     * Build ORU^R01 message for vital signs
     */
    private String buildORUR01Message(String patientId, Map<String, Object> vitalSigns) {
        StringBuilder hl7 = new StringBuilder();
        String timestamp = getCurrentTimestamp();
        
        // MSH segment
        hl7.append("MSH|^~\\&|VitalStream|Hospital|Meditech|Hospital|")
           .append(timestamp)
           .append("||");
        hl7.append("ORU^R01|MSG").append(System.currentTimeMillis()).append("|P|2.5\r");
        
        // PID segment
        hl7.append("PID|1||").append(patientId).append("\r");
        
        // OBR segment
        hl7.append("OBR|1|||").append("VITALS^Vital Signs\r");
        
        // OBX segments for each vital sign
        int seq = 1;
        for (Map.Entry<String, Object> entry : vitalSigns.entrySet()) {
            hl7.append("OBX|").append(seq++).append("|NM|");
            hl7.append(entry.getKey()).append("|");
            hl7.append("|").append(entry.getValue()).append("\r");
        }
        
        return hl7.toString();
    }

    private String getCurrentTimestamp() {
        return new java.text.SimpleDateFormat("yyyyMMddHHmmss").format(new Date());
    }
}
