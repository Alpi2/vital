/**
 * @file dicom_waveform.h
 * @brief DICOM Waveform Storage for ECG data
 * 
 * Implements DICOM Waveform Storage SOP Class (1.2.840.10008.5.1.4.1.1.9.1.1)
 * for storing ECG waveforms in DICOM format.
 * 
 * @author VitalStream Development Team
 * @date 2026-01-03
 * @version 1.0.0
 */

#ifndef VITALSTREAM_DICOM_WAVEFORM_H
#define VITALSTREAM_DICOM_WAVEFORM_H

#ifdef HAVE_DCMTK
#include <dcmtk/dcmdata/dcdatset.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#else
// Minimal fallbacks so this header compiles without DCMTK
class DcmDataset;
#endif

#include <vector>
#include <string>
#include <cstdint>

namespace vitalstream {
namespace dicom {

/**
 * @brief ECG lead type
 */
enum class ECGLead {
    I, II, III,           // Limb leads
    aVR, aVL, aVF,        // Augmented leads
    V1, V2, V3, V4, V5, V6 // Precordial leads
};

/**
 * @brief ECG waveform channel
 */
struct WaveformChannel {
    ECGLead lead;                    ///< Lead type
    std::vector<int16_t> samples;    ///< Sample data
    double sensitivity;              ///< Sensitivity (mV/unit)
    double baseline;                 ///< Baseline value
    std::string label;               ///< Channel label
};

/**
 * @brief ECG waveform data
 */
struct ECGWaveform {
    // Patient information
    std::string patient_id;
    std::string patient_name;
    std::string patient_dob;         // YYYYMMDD
    std::string patient_sex;         // M/F/O
    
    // Study information
    std::string study_instance_uid;
    std::string study_date;          // YYYYMMDD
    std::string study_time;          // HHMMSS
    std::string accession_number;
    
    // Series information
    std::string series_instance_uid;
    std::string series_number;
    std::string modality = "ECG";
    
    // Waveform information
    std::vector<WaveformChannel> channels;
    double sampling_frequency;       // Hz
    uint32_t num_samples;            // Samples per channel
    std::string acquisition_datetime;
    
    // Measurements (optional)
    int heart_rate = 0;              // bpm
    int pr_interval = 0;             // ms
    int qrs_duration = 0;            // ms
    int qt_interval = 0;             // ms
    int qtc_interval = 0;            // ms (corrected)
    int p_axis = 0;                  // degrees
    int qrs_axis = 0;                // degrees
    int t_axis = 0;                  // degrees
    
    // Interpretation (optional)
    std::string interpretation;
};

/**
 * @brief DICOM Waveform converter
 * 
 * Converts VitalStream ECG data to DICOM Waveform format.
 */
class DicomWaveformConverter {
public:
    /**
     * @brief Constructor
     */
    DicomWaveformConverter() = default;

    /**
     * @brief Convert ECG waveform to DICOM dataset
     * @param waveform ECG waveform data
     * @return DICOM dataset (caller owns the pointer)
     */
    DcmDataset* convertToDicom(const ECGWaveform& waveform);

    /**
     * @brief Convert ECG waveform to DICOM file
     * @param waveform ECG waveform data
     * @param filename Output filename
     * @return true if successful
     */
    bool saveToDicomFile(const ECGWaveform& waveform, const std::string& filename);

    /**
     * @brief Load ECG waveform from DICOM file
     * @param filename Input filename
     * @return ECG waveform data
     */
    ECGWaveform loadFromDicomFile(const std::string& filename);

    /**
     * @brief Validate DICOM waveform dataset
     * @param dataset DICOM dataset
     * @return true if valid
     */
    bool validateWaveform(DcmDataset* dataset);

private:
    /**
     * @brief Add patient module
     */
    void addPatientModule(DcmDataset* dataset, const ECGWaveform& waveform);

    /**
     * @brief Add study module
     */
    void addStudyModule(DcmDataset* dataset, const ECGWaveform& waveform);

    /**
     * @brief Add series module
     */
    void addSeriesModule(DcmDataset* dataset, const ECGWaveform& waveform);

    /**
     * @brief Add waveform module
     */
    void addWaveformModule(DcmDataset* dataset, const ECGWaveform& waveform);

    /**
     * @brief Add measurements module
     */
    void addMeasurementsModule(DcmDataset* dataset, const ECGWaveform& waveform);

    /**
     * @brief Generate UID
     */
    std::string generateUID();

    /**
     * @brief Get lead name
     */
    std::string getLeadName(ECGLead lead);

    /**
     * @brief Get lead code
     */
    std::string getLeadCode(ECGLead lead);
};

/**
 * @brief 12-lead ECG builder
 * 
 * Helper class to build standard 12-lead ECG waveforms.
 */
class TwelveLeadECGBuilder {
public:
    /**
     * @brief Constructor
     */
    TwelveLeadECGBuilder();

    /**
     * @brief Set patient information
     */
    TwelveLeadECGBuilder& setPatient(
        const std::string& id,
        const std::string& name,
        const std::string& dob,
        const std::string& sex
    );

    /**
     * @brief Set study information
     */
    TwelveLeadECGBuilder& setStudy(
        const std::string& study_uid,
        const std::string& date,
        const std::string& time
    );

    /**
     * @brief Set sampling frequency
     */
    TwelveLeadECGBuilder& setSamplingFrequency(double freq);

    /**
     * @brief Add lead data
     */
    TwelveLeadECGBuilder& addLead(
        ECGLead lead,
        const std::vector<int16_t>& samples,
        double sensitivity = 0.005  // 5 µV/unit
    );

    /**
     * @brief Set measurements
     */
    TwelveLeadECGBuilder& setMeasurements(
        int hr, int pr, int qrs, int qt, int qtc
    );

    /**
     * @brief Set interpretation
     */
    TwelveLeadECGBuilder& setInterpretation(const std::string& text);

    /**
     * @brief Build ECG waveform
     */
    ECGWaveform build();

private:
    ECGWaveform waveform_;
};

} // namespace dicom
} // namespace vitalstream

#endif // VITALSTREAM_DICOM_WAVEFORM_H
