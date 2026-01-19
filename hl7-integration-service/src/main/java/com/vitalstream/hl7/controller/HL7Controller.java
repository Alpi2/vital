package com.vitalstream.hl7.controller;

import ca.uhn.hl7v2.HL7Exception;
import ca.uhn.hl7v2.model.Message;
import java.io.IOException;
import ca.uhn.hl7v2.model.v25.message.ORU_R01;
import com.vitalstream.hl7.service.HL7Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * HL7 REST API Controller
 */
@RestController
@RequestMapping("/api/hl7")
public class HL7Controller {

    private static final Logger log = LoggerFactory.getLogger(HL7Controller.class);
    private final HL7Service hl7Service;

    public HL7Controller(HL7Service hl7Service) {
        this.hl7Service = hl7Service;
    }

    @PostMapping("/parse")
    public ResponseEntity<String> parseMessage(@RequestBody String messageString) {
        try {
            Message message = hl7Service.parseMessage(messageString);
            String encoded = hl7Service.encodeMessage(message);
            return ResponseEntity.ok(encoded);
        } catch (HL7Exception e) {
            log.error("Failed to parse HL7 message", e);
            return ResponseEntity.badRequest().body("Invalid HL7 message: " + e.getMessage());
        }
    }

    @PostMapping("/observation")
    public ResponseEntity<String> createObservation(
            @RequestParam String patientId,
            @RequestParam String type,
            @RequestParam String value,
            @RequestParam String unit) {
        try {
            ORU_R01 message = hl7Service.createObservationMessage(
                patientId, type, value, unit);
            String encoded = hl7Service.encodeMessage(message);
            return ResponseEntity.ok(encoded);
        } catch (HL7Exception | IOException e) {
            log.error("Failed to create observation message", e);
            return ResponseEntity.internalServerError()
                .body("Failed to create message: " + e.getMessage());
        }
    }

    @PostMapping("/validate")
    public ResponseEntity<Boolean> validateMessage(@RequestBody String messageString) {
        boolean isValid = hl7Service.validateMessage(messageString);
        return ResponseEntity.ok(isValid);
    }
}
