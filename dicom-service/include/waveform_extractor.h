#pragma once

#include <optional>
#include <vector>
#include <cstdint>
#include "dicom_service.pb.h"

// Forward declarations for DCMTK
class DcmDataset;
class DcmItem;

namespace vitalstream {
namespace dicom {

/**
 * Extract calibrated ECG waveform data from DICOM dataset
 * 
 * This function handles the complete extraction process including:
 * - Finding waveform sequences in DICOM data
 * - Extracting multi-channel waveform data
 * - Applying calibration (sensitivity and baseline)
 * Converting to standardized float values
 * 
 * @param dataset DCMTK dataset containing waveform data
 * @return Optional WaveformData structure with calibrated values
 */
std::optional<::dicom::v1::WaveformData> extract_ecg_waveform(DcmDataset* dataset);

/**
 * Validate extracted waveform data for medical accuracy
 * 
 * @param waveform Extracted waveform data
 * @return true if waveform data is medically valid
 */
bool validate_waveform_medical_accuracy(const ::dicom::v1::WaveformData& waveform);

/**
 * Apply DICOM calibration to raw waveform values
 * 
 * Formula: calibrated_value = (raw_value * sensitivity) + baseline
 * 
 * @param raw_value Raw ADC value from DICOM
 * @param sensitivity Channel sensitivity factor
 * @param baseline Channel baseline offset
 * @return Calibrated value in medical units (typically mV)
 */
float apply_dicom_calibration(uint16_t raw_value, double sensitivity, double baseline);

/**
 * Extract channel-specific metadata from DICOM waveform sequence
 * 
 * @param item DCMTK item containing channel data
 * @param channel_name Output parameter for channel name
 * @param sensitivity Output parameter for sensitivity
 * @param baseline Output parameter for baseline
 * @return true if channel metadata extracted successfully
 */
bool extract_channel_metadata(DcmItem* item, 
                             std::string& channel_name,
                             double& sensitivity, 
                             double& baseline);

/**
 * Convert multi-channel interleaved waveform data to separate channels
 * 
 * DICOM stores multi-channel data in interleaved format:
 * [ch1_sample1, ch2_sample1, ch3_sample1, ch1_sample2, ch2_sample2, ch3_sample2, ...]
 * 
 * @param interleaved_data Raw interleaved waveform data
 * @param num_channels Number of channels
 * @param num_samples Number of samples per channel
 * @return Vector of vectors, one per channel
 */
std::vector<std::vector<float>> deinterleave_waveform(
    const std::vector<uint16_t>& interleaved_data,
    int32_t num_channels,
    int32_t num_samples
);

} // namespace dicom
} // namespace vitalstream
