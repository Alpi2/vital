package com.vitalstream.hl7.controller;

import com.vitalstream.hl7.service.FHIRService;
import org.hl7.fhir.r4.model.Observation;
import org.hl7.fhir.r4.model.Patient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Date;

/**
 * FHIR REST API Controller
 */
@RestController
@RequestMapping("/api/fhir")
public class FHIRController {

    private static final Logger log = LoggerFactory.getLogger(FHIRController.class);
    private final FHIRService fhirService;

    public FHIRController(FHIRService fhirService) {
        this.fhirService = fhirService;
    }

    @PostMapping("/observation")
    public ResponseEntity<String> createObservation(
            @RequestParam String patientId,
            @RequestParam String code,
            @RequestParam String display,
            @RequestParam double value,
            @RequestParam String unit) {
        
        Observation observation = fhirService.createVitalSignObservation(
            patientId, code, display, value, unit);
        
        String json = fhirService.encodeToJson(observation);
        return ResponseEntity.ok(json);
    }

    @PostMapping("/patient")
    public ResponseEntity<String> createPatient(
            @RequestParam String patientId,
            @RequestParam String familyName,
            @RequestParam String givenName,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd") Date birthDate) {
        
        Patient patient = fhirService.createPatient(
            patientId, familyName, givenName, birthDate);
        
        String json = fhirService.encodeToJson(patient);
        return ResponseEntity.ok(json);
    }

    @PostMapping("/observation/submit")
    public ResponseEntity<String> submitObservation(@RequestBody String observationJson) {
        try {
            Observation observation = fhirService.parseJson(
                observationJson, Observation.class);
            String id = fhirService.submitObservation(observation);
            return ResponseEntity.ok("Observation submitted with ID: " + id);
        } catch (Exception e) {
            log.error("Failed to submit observation", e);
            return ResponseEntity.internalServerError()
                .body("Failed to submit: " + e.getMessage());
        }
    }
}
