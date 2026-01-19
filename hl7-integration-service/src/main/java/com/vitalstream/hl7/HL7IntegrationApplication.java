package com.vitalstream.hl7;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * VitalStream HL7/FHIR/DICOM Integration Service
 * 
 * FDA/IEC 62304 compliant medical data integration service
 * Provides enterprise-grade HL7, FHIR, and DICOM support
 */
@SpringBootApplication
public class HL7IntegrationApplication {

    public static void main(String[] args) {
        SpringApplication.run(HL7IntegrationApplication.class, args);
    }
}
