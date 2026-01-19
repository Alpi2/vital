package com.vitalstream.hl7.service;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.parser.IParser;
import ca.uhn.fhir.rest.client.api.IGenericClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.hl7.fhir.r4.model.*;
import ca.uhn.fhir.rest.api.MethodOutcome;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Date;

/**
 * FHIR R4 Service
 * 
 * Handles FHIR resource creation, parsing, and server communication
 * Uses HAPI FHIR library for enterprise-grade FHIR support
 */
@Service
public class FHIRService {

    private static final Logger log = LoggerFactory.getLogger(FHIRService.class);
    private final FhirContext fhirContext;
    private final IParser jsonParser;
    private final IParser xmlParser;
    
    @Value("${fhir.server.url:http://localhost:8080/fhir}")
    private String fhirServerUrl;

    public FHIRService() {
        this.fhirContext = FhirContext.forR4();
        this.jsonParser = fhirContext.newJsonParser().setPrettyPrint(true);
        this.xmlParser = fhirContext.newXmlParser().setPrettyPrint(true);
    }

    /**
     * Create FHIR client
     */
    public IGenericClient createClient() {
        return fhirContext.newRestfulGenericClient(fhirServerUrl);
    }

    /**
     * Create Observation resource for vital signs
     */
    public Observation createVitalSignObservation(
            String patientId,
            String code,
            String display,
            double value,
            String unit) {
        
        log.info("Creating FHIR Observation for patient: {}", patientId);
        
        Observation observation = new Observation();
        
        // Status
        observation.setStatus(Observation.ObservationStatus.FINAL);
        
        // Category - vital signs
        CodeableConcept category = new CodeableConcept();
        category.addCoding()
            .setSystem("http://terminology.hl7.org/CodeSys tem/observation-category")
            .setCode("vital-signs")
            .setDisplay("Vital Signs");
        observation.addCategory(category);
        
        // Code - what was observed
        CodeableConcept codeableConcept = new CodeableConcept();
        codeableConcept.addCoding()
            .setSystem("http://loinc.org")
            .setCode(code)
            .setDisplay(display);
        observation.setCode(codeableConcept);
        
        // Subject - patient reference
        observation.setSubject(new Reference("Patient/" + patientId));
        
        // Effective time
        observation.setEffective(new DateTimeType(new Date()));
        
        // Value
        Quantity quantity = new Quantity();
        quantity.setValue(value);
        quantity.setUnit(unit);
        quantity.setSystem("http://unitsofmeasure.org");
        quantity.setCode(unit);
        observation.setValue(quantity);
        
        log.info("FHIR Observation created successfully");
        return observation;
    }

    /**
     * Create Patient resource
     */
    public Patient createPatient(
            String patientId,
            String familyName,
            String givenName,
            Date birthDate) {
        
        log.info("Creating FHIR Patient: {}", patientId);
        
        Patient patient = new Patient();
        
        // Identifier
        patient.addIdentifier()
            .setSystem("http://vitalstream.com/patient-id")
            .setValue(patientId);
        
        // Name
        HumanName name = new HumanName();
        name.setFamily(familyName);
        name.addGiven(givenName);
        patient.addName(name);
        
        // Birth date
        patient.setBirthDate(birthDate);
        
        // Active
        patient.setActive(true);
        
        log.info("FHIR Patient created successfully");
        return patient;
    }

    /**
     * Parse FHIR resource from JSON
     */
    public <T extends Resource> T parseJson(String json, Class<T> resourceType) {
        return jsonParser.parseResource(resourceType, json);
    }

    /**
     * Parse FHIR resource from XML
     */
    public <T extends Resource> T parseXml(String xml, Class<T> resourceType) {
        return xmlParser.parseResource(resourceType, xml);
    }

    /**
     * Encode resource to JSON
     */
    public String encodeToJson(Resource resource) {
        return jsonParser.encodeResourceToString(resource);
    }

    /**
     * Encode resource to XML
     */
    public String encodeToXml(Resource resource) {
        return xmlParser.encodeResourceToString(resource);
    }

    /**
     * Submit observation to FHIR server
     */
    public String submitObservation(Observation observation) {
        IGenericClient client = createClient();
        
        MethodOutcome outcome = client.create()
            .resource(observation)
            .execute();
        
        String id = outcome.getId().getIdPart();
        log.info("Observation submitted to FHIR server with ID: {}", id);
        
        return id;
    }
}
