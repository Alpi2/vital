package com.vitalstream.hl7.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.dcm4che3.data.Attributes;
import org.dcm4che3.data.Tag;
import org.dcm4che3.data.UID;
import org.dcm4che3.data.VR;
import org.dcm4che3.io.DicomOutputStream;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * DICOM Service
 * 
 * Handles DICOM object creation and storage for ECG waveforms
 * Uses DCM4CHE library for DICOM support
 */
@Service
public class DICOMService {

    private static final Logger log = LoggerFactory.getLogger(DICOMService.class);
    private static final DateTimeFormatter DICOM_DATE_FORMAT = 
        DateTimeFormatter.ofPattern("yyyyMMdd");
    private static final DateTimeFormatter DICOM_TIME_FORMAT = 
        DateTimeFormatter.ofPattern("HHmmss");

    /**
     * Create DICOM ECG object
     */
    public Attributes createECGObject(
            String patientId,
            String patientName,
            int samplingRate,
            int numberOfChannels,
            short[][] waveformData) {
        
        log.info("Creating DICOM ECG object for patient: {}", patientId);
        
        Attributes attrs = new Attributes();
        
        // SOP Common Module
        attrs.setString(Tag.SOPClassUID, VR.UI, UID.TwelveLeadECGWaveformStorage);
        attrs.setString(Tag.SOPInstanceUID, VR.UI, UUID.randomUUID().toString());
        
        // Patient Module
        attrs.setString(Tag.PatientID, VR.LO, patientId);
        attrs.setString(Tag.PatientName, VR.PN, patientName);
        
        // Study Module
        attrs.setString(Tag.StudyInstanceUID, VR.UI, UUID.randomUUID().toString());
        attrs.setString(Tag.StudyDate, VR.DA, 
            LocalDateTime.now().format(DICOM_DATE_FORMAT));
        attrs.setString(Tag.StudyTime, VR.TM, 
            LocalDateTime.now().format(DICOM_TIME_FORMAT));
        
        // Series Module
        attrs.setString(Tag.SeriesInstanceUID, VR.UI, UUID.randomUUID().toString());
        attrs.setString(Tag.Modality, VR.CS, "ECG");
        attrs.setInt(Tag.SeriesNumber, VR.IS, 1);
        
        // Waveform Module
        attrs.setInt(Tag.SamplingFrequency, VR.DS, samplingRate);
        attrs.setInt(Tag.NumberOfWaveformChannels, VR.US, numberOfChannels);
        attrs.setInt(Tag.NumberOfWaveformSamples, VR.UL, 
            waveformData.length > 0 ? waveformData[0].length : 0);
        
        // Waveform data would be added here
        // This is a simplified version
        
        log.info("DICOM ECG object created successfully");
        return attrs;
    }

    /**
     * Save DICOM object to file
     */
    public void saveDICOMFile(Attributes attrs, File file) throws IOException {
        log.info("Saving DICOM file: {}", file.getAbsolutePath());
        
        try (DicomOutputStream dos = new DicomOutputStream(file)) {
            dos.writeDataset(attrs.createFileMetaInformation(
                UID.ExplicitVRLittleEndian), attrs);
        }
        
        log.info("DICOM file saved successfully");
    }

    /**
     * Create DICOM query attributes
     */
    public Attributes createQueryAttributes(String patientId) {
        Attributes attrs = new Attributes();
        attrs.setString(Tag.QueryRetrieveLevel, VR.CS, "PATIENT");
        attrs.setString(Tag.PatientID, VR.LO, patientId);
        return attrs;
    }
}
