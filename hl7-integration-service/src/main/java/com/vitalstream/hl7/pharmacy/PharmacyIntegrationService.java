package com.vitalstream.hl7.pharmacy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Pharmacy Integration Service
 * 
 * Handles integration with pharmacy systems for:
 * - Drug information
 * - Drug-drug interactions
 * - QT prolongation risk assessment
 * - Medication Administration Record (MAR)
 * 
 * @author VitalStream Development Team
 * @version 1.0.0
 * @since 2026-01-03
 */
@Service
public class PharmacyIntegrationService {

    private static final Logger log = LoggerFactory.getLogger(PharmacyIntegrationService.class);

    /**
     * Drug information
     */
    public static class DrugInfo {
        private String drugCode;      // RxNorm or ATC code
        private String drugName;
        private String genericName;
        private String brandName;
        private String dosageForm;    // Tablet, Capsule, Injection, etc.
        private String strength;
        private String route;         // PO, IV, IM, SC, etc.
        private boolean qtProlonging; // QT prolongation risk
        private String category;      // Therapeutic category
        
        // Getters and setters
        public String getDrugCode() { return drugCode; }
        public void setDrugCode(String drugCode) { this.drugCode = drugCode; }
        
        public String getDrugName() { return drugName; }
        public void setDrugName(String drugName) { this.drugName = drugName; }
        
        public String getGenericName() { return genericName; }
        public void setGenericName(String genericName) { this.genericName = genericName; }
        
        public String getBrandName() { return brandName; }
        public void setBrandName(String brandName) { this.brandName = brandName; }
        
        public String getDosageForm() { return dosageForm; }
        public void setDosageForm(String dosageForm) { this.dosageForm = dosageForm; }
        
        public String getStrength() { return strength; }
        public void setStrength(String strength) { this.strength = strength; }
        
        public String getRoute() { return route; }
        public void setRoute(String route) { this.route = route; }
        
        public boolean isQtProlonging() { return qtProlonging; }
        public void setQtProlonging(boolean qtProlonging) { this.qtProlonging = qtProlonging; }
        
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
    }

    /**
     * Drug interaction
     */
    public static class DrugInteraction {
        public enum Severity {
            MINOR,          // Minimal clinical significance
            MODERATE,       // May require monitoring
            MAJOR,          // Avoid combination if possible
            CONTRAINDICATED // Do not use together
        }
        
        private String drug1Code;
        private String drug1Name;
        private String drug2Code;
        private String drug2Name;
        private Severity severity;
        private String description;
        private String recommendation;
        
        // Getters and setters
        public String getDrug1Code() { return drug1Code; }
        public void setDrug1Code(String drug1Code) { this.drug1Code = drug1Code; }
        
        public String getDrug1Name() { return drug1Name; }
        public void setDrug1Name(String drug1Name) { this.drug1Name = drug1Name; }
        
        public String getDrug2Code() { return drug2Code; }
        public void setDrug2Code(String drug2Code) { this.drug2Code = drug2Code; }
        
        public String getDrug2Name() { return drug2Name; }
        public void setDrug2Name(String drug2Name) { this.drug2Name = drug2Name; }
        
        public Severity getSeverity() { return severity; }
        public void setSeverity(Severity severity) { this.severity = severity; }
        
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        
        public String getRecommendation() { return recommendation; }
        public void setRecommendation(String recommendation) { this.recommendation = recommendation; }
    }

    /**
     * Medication administration record
     */
    public static class MedicationAdministration {
        private String patientId;
        private String drugCode;
        private String drugName;
        private String dose;
        private String route;
        private Date scheduledTime;
        private Date administeredTime;
        private String administeredBy;  // Nurse/practitioner ID
        private String status;          // Scheduled, Given, Held, Refused
        private String reason;          // Reason if held/refused
        
        // Getters and setters
        public String getPatientId() { return patientId; }
        public void setPatientId(String patientId) { this.patientId = patientId; }
        
        public String getDrugCode() { return drugCode; }
        public void setDrugCode(String drugCode) { this.drugCode = drugCode; }
        
        public String getDrugName() { return drugName; }
        public void setDrugName(String drugName) { this.drugName = drugName; }
        
        public String getDose() { return dose; }
        public void setDose(String dose) { this.dose = dose; }
        
        public String getRoute() { return route; }
        public void setRoute(String route) { this.route = route; }
        
        public Date getScheduledTime() { return scheduledTime; }
        public void setScheduledTime(Date scheduledTime) { this.scheduledTime = scheduledTime; }
        
        public Date getAdministeredTime() { return administeredTime; }
        public void setAdministeredTime(Date administeredTime) { this.administeredTime = administeredTime; }
        
        public String getAdministeredBy() { return administeredBy; }
        public void setAdministeredBy(String administeredBy) { this.administeredBy = administeredBy; }
        
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
    }

    // QT-prolonging drugs database (simplified)
    private static final Set<String> QT_PROLONGING_DRUGS = Set.of(
        "AMIODARONE", "SOTALOL", "DOFETILIDE", "IBUTILIDE",
        "AZITHROMYCIN", "CLARITHROMYCIN", "ERYTHROMYCIN",
        "HALOPERIDOL", "DROPERIDOL", "CHLORPROMAZINE",
        "METHADONE", "ONDANSETRON", "DOMPERIDONE"
    );

    /**
     * Check drug-drug interactions
     * 
     * @param drugCodes List of drug codes
     * @return List of interactions
     */
    public List<DrugInteraction> checkDrugInteractions(List<String> drugCodes) {
        List<DrugInteraction> interactions = new ArrayList<>();
        
        // Check all pairs
        for (int i = 0; i < drugCodes.size(); i++) {
            for (int j = i + 1; j < drugCodes.size(); j++) {
                DrugInteraction interaction = checkInteraction(drugCodes.get(i), drugCodes.get(j));
                if (interaction != null) {
                    interactions.add(interaction);
                }
            }
        }
        
        log.info("Found {} drug interactions for {} drugs", interactions.size(), drugCodes.size());
        return interactions;
    }

    /**
     * Check QT prolongation risk
     * 
     * @param drugCodes List of drug codes
     * @param currentQTc Current QTc interval (ms)
     * @return List of QT-prolonging drugs
     */
    public List<DrugInfo> checkQTProlongationRisk(List<String> drugCodes, int currentQTc) {
        List<DrugInfo> qtDrugs = new ArrayList<>();
        
        for (String code : drugCodes) {
            DrugInfo drug = getDrugInfo(code);
            if (drug != null && drug.isQtProlonging()) {
                qtDrugs.add(drug);
            }
        }
        
        if (!qtDrugs.isEmpty()) {
            log.warn("Patient has {} QT-prolonging drugs with QTc = {} ms",
                qtDrugs.size(), currentQTc);
            
            if (currentQTc > 500) {
                log.error("CRITICAL: QTc > 500 ms with QT-prolonging drugs!");
            }
        }
        
        return qtDrugs;
    }

    /**
     * Get drug information
     * 
     * @param drugCode Drug code (RxNorm/ATC)
     * @return Drug information
     */
    public DrugInfo getDrugInfo(String drugCode) {
        // Implementation pending: actual drug database lookup
        // This is a placeholder implementation
        
        DrugInfo drug = new DrugInfo();
        drug.setDrugCode(drugCode);
        drug.setDrugName(drugCode); // Placeholder
        drug.setQtProlonging(QT_PROLONGING_DRUGS.contains(drugCode.toUpperCase()));
        
        return drug;
    }

    /**
     * Get medication administration records
     * 
     * @param patientId Patient ID
     * @param startDate Start date
     * @param endDate End date
     * @return List of medication administrations
     */
    public List<MedicationAdministration> getMedicationAdministrations(
            String patientId, Date startDate, Date endDate) {
        // Implementation pending: MAR retrieval from pharmacy system
        
        log.info("Retrieving MAR for patient {} from {} to {}",
            patientId, startDate, endDate);
        
        return new ArrayList<>();
    }

    /**
     * Record medication administration
     * 
     * @param administration Medication administration record
     * @return true if successful
     */
    public boolean recordAdministration(MedicationAdministration administration) {
        // Implementation pending: MAR recording
        log.info("Recording medication administration: {} for patient {}",
            administration.getDrugName(), administration.getPatientId());
        
        // Validate five rights
        if (!validateFiveRights(administration)) {
            log.error("Five rights validation failed!");
            return false;
        }
        
        return true;
    }

    /**
     * Validate five rights of medication administration
     * - Right patient
     * - Right drug
     * - Right dose
     * - Right route
     * - Right time
     */
    private boolean validateFiveRights(MedicationAdministration admin) {
        // Implementation pending: actual validation
        return admin.getPatientId() != null &&
               admin.getDrugCode() != null &&
               admin.getDose() != null &&
               admin.getRoute() != null &&
               admin.getScheduledTime() != null;
    }

    /**
     * Check interaction between two drugs
     */
    private DrugInteraction checkInteraction(String drug1Code, String drug2Code) {
        // Implementation pending: actual interaction checking
        // This is a placeholder that checks for known dangerous combinations
        
        String d1 = drug1Code.toUpperCase();
        String d2 = drug2Code.toUpperCase();
        
        // Example: Warfarin + NSAIDs
        if ((d1.contains("WARFARIN") && d2.contains("IBUPROFEN")) ||
            (d2.contains("WARFARIN") && d1.contains("IBUPROFEN"))) {
            
            DrugInteraction interaction = new DrugInteraction();
            interaction.setDrug1Code(drug1Code);
            interaction.setDrug1Name("Warfarin");
            interaction.setDrug2Code(drug2Code);
            interaction.setDrug2Name("Ibuprofen");
            interaction.setSeverity(DrugInteraction.Severity.MAJOR);
            interaction.setDescription("Increased risk of bleeding");
            interaction.setRecommendation("Monitor INR closely. Consider alternative analgesic.");
            
            return interaction;
        }
        
        return null;
    }
}
