#include "waveform_extractor.h"
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcdatset.h>
#include <dcmtk/dcmdata/dcitem.h>
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace vitalstream {
namespace dicom {

// Helper function for DICOM calibration
float apply_dicom_calibration(uint16_t raw_value, double sensitivity, double baseline) {
    // DICOM calibration formula: calibrated_value = (raw_value * sensitivity) + baseline
    // This converts raw ADC values to medical units (typically mV for ECG)
    return static_cast<float>(raw_value * sensitivity + baseline);
}

    spdlog::info("🔍 Extracting ECG waveform from DICOM dataset");
    
    if (!dataset) {
        spdlog::error("❌ Null dataset provided to waveform extractor");
        return std::nullopt;
    }
    
    // Find waveform sequence
    DcmElement* element = nullptr;
    OFCondition status = dataset->findAndGetElement(DCM_WaveformSequence, element);
    
    if (status.bad() || !element) {
        spdlog::warn("⚠️ No waveform sequence found in DICOM dataset");
        return std::nullopt;
    }
    
    DcmSequenceOfItems* sequence = dynamic_cast<DcmSequenceOfItems*>(element);
    if (!sequence || sequence->card() == 0) {
        spdlog::warn("⚠️ Empty waveform sequence found");
        return std::nullopt;
    }
    
    spdlog::info("📊 Found waveform sequence with {} items", sequence->card());
    
    // Initialize waveform structure
    ::dicom::v1::WaveformData waveform;
    std::vector<std::vector<float>> channel_data;
    std::vector<std::string> global_channel_names;
    
    // Variables to store metadata from first item
    Uint16 totalChannels = 0;
    Uint32 totalSamples = 0;
    Float64 overallSamplingFrequency = 0.0;
    
    // Process each waveform item (typically one per channel group)
    for (size_t item_idx = 0; item_idx < sequence->card(); ++item_idx) {
        DcmItem* item = sequence->getItem(item_idx);
        if (!item) {
            spdlog::warn("⚠️ Could not get waveform item {}", item_idx);
            continue;
        }
        
        spdlog::debug("Processing waveform item {}", item_idx);
        
        // Extract waveform metadata
        Uint16 numChannels = 0;
        Uint32 numSamples = 0;
        Float64 samplingFrequency = 0.0;
        
        status = item->findAndGetUint16(DCM_NumberOfWaveformChannels, numChannels);
        if (status.bad()) {
            spdlog::warn("⚠️ Could not get number of channels for item {}", item_idx);
            continue;
        }
        
        status = item->findAndGetUint32(DCM_NumberOfWaveformSamples, numSamples);
        if (status.bad()) {
            spdlog::warn("⚠️ Could not get number of samples for item {}", item_idx);
            continue;
        }
        
        status = item->findAndGetFloat64(DCM_SamplingFrequency, samplingFrequency);
        if (status.bad()) {
            spdlog::warn("⚠️ Could not get sampling frequency for item {}", item_idx);
            samplingFrequency = 250.0; // Default ECG sampling rate
        }
        
        // Store metadata from first item
        if (item_idx == 0) {
            totalChannels = numChannels;
            totalSamples = numSamples;
            overallSamplingFrequency = samplingFrequency;
            
            // Set waveform metadata
            waveform.set_num_channels(totalChannels);
            waveform.set_num_samples(totalSamples);
            waveform.set_sampling_frequency(overallSamplingFrequency);
            
            // Initialize channel data vectors
            channel_data.resize(totalChannels);
            for (int ch = 0; ch < totalChannels; ++ch) {
                channel_data[ch].reserve(totalSamples);
            }
        }
        
        // Extract channel-specific calibration parameters
        std::vector<double> sensitivities(numChannels, 1.0);
        std::vector<double> baselines(numChannels, 0.0);
        std::vector<std::string> local_channel_names(numChannels);
        
        for (int ch = 0; ch < numChannels; ++ch) {
            // Get channel sensitivity
            DcmElement* sensElement = nullptr;
            if (item->findAndGetElement(DCM_ChannelSensitivity, sensElement).good() && sensElement) {
                DcmSequenceOfItems* sensSequence = dynamic_cast<DcmSequenceOfItems*>(sensElement);
                if (sensSequence && sensSequence->card() > ch) {
                    DcmItem* sensItem = sensSequence->getItem(ch);
                    Float64 sensitivity = 1.0;
                    if (sensItem->findAndGetFloat64(DCM_ChannelSensitivity, sensitivity).good()) {
                        sensitivities[ch] = sensitivity;
                    }
                }
            }
            
            // Get channel baseline
            DcmElement* baselineElement = nullptr;
            if (item->findAndGetElement(DCM_ChannelBaseline, baselineElement).good() && baselineElement) {
                DcmSequenceOfItems* baselineSequence = dynamic_cast<DcmSequenceOfItems*>(baselineElement);
                if (baselineSequence && baselineSequence->card() > ch) {
                    DcmItem* baselineItem = baselineSequence->getItem(ch);
                    Float64 baseline = 0.0;
                    if (baselineItem->findAndGetFloat64(DCM_ChannelBaseline, baseline).good()) {
                        baselines[ch] = baseline;
                    }
                }
            }
            
            // Get channel name
            DcmElement* labelElement = nullptr;
            if (item->findAndGetElement(DCM_ChannelLabel, labelElement).good() && labelElement) {
                DcmSequenceOfItems* labelSequence = dynamic_cast<DcmSequenceOfItems*>(labelElement);
                if (labelSequence && labelSequence->card() > ch) {
                    DcmItem* labelItem = labelSequence->getItem(ch);
                    OFString label;
                    if (labelItem->findAndGetOFString(DCM_ChannelLabel, label).good()) {
                        local_channel_names[ch] = std::string(label.c_str());
                    } else {
                        local_channel_names[ch] = "Channel_" + std::to_string(ch + 1);
                    }
                }
            } else {
                local_channel_names[ch] = "Channel_" + std::to_string(ch + 1);
            }
        }
        
        // Add local channel names to global list
        global_channel_names.insert(global_channel_names.end(), local_channel_names.begin(), local_channel_names.end());
        
        // Get waveform data
        const Uint16* waveformData = nullptr;
        unsigned long count = 0;
        status = item->findAndGetUint16Array(DCM_WaveformData, waveformData, &count);
        
        if (status.bad() || !waveformData) {
            spdlog::warn("⚠️ No waveform data found for item {}", item_idx);
            continue;
        }
        
        spdlog::info("📈 Extracting {} samples for {} channels", count, numChannels);
        
        // Verify data size
        if (count != numChannels * numSamples) {
            spdlog::warn("⚠️ Data size mismatch: expected {}, got {}", 
                        numChannels * numSamples, count);
            continue;
        }
        
        // Extract and calibrate samples for each channel
        for (unsigned long sample_idx = 0; sample_idx < count; ++sample_idx) {
            int channel_idx = sample_idx % numChannels;
            int sample_in_channel = sample_idx / numChannels;
            
            if (sample_in_channel >= numSamples) break;
            
            // Apply DICOM calibration: calibrated = (raw * sensitivity) + baseline
            float calibrated_value = apply_dicom_calibration(
                waveformData[sample_idx],
                sensitivities[channel_idx],
                baselines[channel_idx]
            );
            
            channel_data[channel_idx].push_back(calibrated_value);
        }
        
        // Store calibration info from first item
        if (item_idx == 0) {
            waveform.set_sensitivity(sensitivities[0]); // Use first channel as representative
            waveform.set_baseline(baselines[0]);
            waveform.set_units("mV"); // Standard ECG units
            
            // Extract bit depth information
            Uint16 bitsAllocated = 16;
            Uint16 bitsStored = 16;
            item->findAndGetUint16(DCM_WaveformBitsAllocated, bitsAllocated);
            item->findAndGetUint16(DCM_WaveformBitsStored, bitsStored);
            
            waveform.set_bits_allocated(bitsAllocated);
            waveform.set_bits_stored(bitsStored);
            waveform.set_is_signed(true); // Assume signed for ECG
            
            spdlog::debug("📊 Waveform metadata: {} channels, {} samples, {:.1f} Hz, {} bits", 
                        totalChannels, totalSamples, overallSamplingFrequency, bitsStored);
        }
    }
    
    // Check if we extracted any data
    if (channel_data.empty() || channel_data[0].empty()) {
        spdlog::warn("⚠️ No valid waveform data extracted");
        return std::nullopt;
    }
    
    // Add channel names to waveform
    for (const auto& name : global_channel_names) {
        waveform.add_channel_names(name);
    }
    
    // Interleave channel data for DICOM format (ch1_s1, ch2_s1, ch3_s1, ch1_s2, ch2_s2, ch3_s2, ...)
    int total_channels = channel_data.size();
    int samples_per_channel = channel_data[0].size();
    
    spdlog::info("🔄 Interleaving {} channels with {} samples each", 
                total_channels, samples_per_channel);
    
    for (int sample_idx = 0; sample_idx < samples_per_channel; ++sample_idx) {
        for (int ch = 0; ch < total_channels; ++ch) {
            if (sample_idx < channel_data[ch].size()) {
                waveform.add_samples(channel_data[ch][sample_idx]);
            }
        }
    }
    
    // Validate medical accuracy
    if (!validate_waveform_medical_accuracy(waveform)) {
        spdlog::warn("⚠️ Extracted waveform failed medical accuracy validation");
        // Still return the data but log the warning
    }
    
    spdlog::info("✅ ECG waveform extracted successfully: {} channels, {} samples, {:.1f} Hz",
                 waveform.num_channels(), waveform.num_samples(), waveform.sampling_frequency());
    
    // Log calibration info
    spdlog::info("📊 Calibration: sensitivity={}, baseline={}, units={}",
                waveform.sensitivity(), waveform.baseline(), waveform.units());
    
    return waveform;
}

bool validate_waveform_medical_accuracy(const ::dicom::v1::WaveformData& waveform) {
    // Check for reasonable ECG parameters
    if (waveform.num_channels() <= 0 || waveform.num_channels() > 16) {
        spdlog::warn("Invalid channel count: {}", waveform.num_channels());
        return false;
    }
    
    if (waveform.num_samples() <= 0 || waveform.num_samples() > 100000) {
        spdlog::warn("Invalid sample count: {}", waveform.num_samples());
        return false;
    }
    
    if (waveform.sampling_frequency() < 100.0 || waveform.sampling_frequency() > 2000.0) {
        spdlog::warn("Invalid sampling frequency: {:.1f} Hz", waveform.sampling_frequency());
        return false;
    }
    
    // Check for reasonable ECG voltage ranges (typically ±5mV)
    if (waveform.samples_size() > 0) {
        float min_val = *std::min_element(waveform.samples().begin(), waveform.samples().end());
        float max_val = *std::max_element(waveform.samples().begin(), waveform.samples().end());
        
        if (std::abs(min_val) > 50.0 || std::abs(max_val) > 50.0) {
            spdlog::warn("Suspicious voltage range: {:.3f} to {:.3f} mV", min_val, max_val);
            return false;
        }
    }
    
    return true;
}

float apply_dicom_calibration(uint16_t raw_value, double sensitivity, double baseline) {
    // DICOM calibration formula: calibrated_value = (raw_value * sensitivity) + baseline
    // This converts raw ADC values to medical units (typically mV for ECG)
    
    // Handle edge cases
    if (sensitivity == 0.0) {
        sensitivity = 1.0; // Prevent division by zero
    }
    
    float calibrated = static_cast<float>((static_cast<double>(raw_value) * sensitivity) + baseline);
    
    // Clamp to reasonable ECG range (±50mV) to prevent extreme values
    calibrated = std::max(-50.0f, std::min(50.0f, calibrated));
    
    return calibrated;
}

bool extract_channel_metadata(DcmItem* item, 
                             std::string& channel_name,
                             double& sensitivity, 
                             double& baseline) {
    if (!item) {
        return false;
    }
    
    // Default values
    sensitivity = 1.0;
    baseline = 0.0;
    channel_name = "";
    
    // Extract channel name
    OFString label;
    if (item->findAndGetOFString(DCM_ChannelLabel, label).good()) {
        channel_name = std::string(label.c_str());
    }
    
    // Extract sensitivity
    Float64 sens = 0.0;
    if (item->findAndGetFloat64(DCM_ChannelSensitivity, sens).good()) {
        sensitivity = static_cast<double>(sens);
    }
    
    // Extract baseline
    Float64 base = 0.0;
    if (item->findAndGetFloat64(DCM_ChannelBaseline, base).good()) {
        baseline = static_cast<double>(base);
    }
    
    return true;
}

std::vector<std::vector<float>> deinterleave_waveform(
    const std::vector<uint16_t>& interleaved_data,
    int32_t num_channels,
    int32_t num_samples
) {
    std::vector<std::vector<float>> channel_data(num_channels);
    
    // Pre-allocate memory for efficiency
    for (int ch = 0; ch < num_channels; ++ch) {
        channel_data[ch].reserve(num_samples);
    }
    
    // Deinterleave: [ch1_s1, ch2_s1, ch3_s1, ch1_s2, ch2_s2, ch3_s2, ...]
    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        for (int ch = 0; ch < num_channels; ++ch) {
            int data_idx = sample_idx * num_channels + ch;
            if (data_idx < interleaved_data.size()) {
                channel_data[ch].push_back(static_cast<float>(interleaved_data[data_idx]));
            }
        }
    }
    
    return channel_data;
}

} // namespace dicom
} // namespace vitalstream
