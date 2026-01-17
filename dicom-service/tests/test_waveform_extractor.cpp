#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include "dicom/v1/dicom_service.pb.h"
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcdatset.h>
#include "../src/waveform_extractor.h"

using namespace vitalstream::dicom;

class WaveformExtractorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test DICOM dataset with waveform data
        test_dataset_ = std::make_unique<DcmDataset>();
        
        // Add basic DICOM tags
        test_dataset_->putAndInsertString(DCM_SOPClassUID, "1.2.840.10008.5.1.4.1.1.2.1");
        test_dataset_->putAndInsertString(DCM_SOPInstanceUID, "1.2.3.4.5.6.7.8.9");
        test_dataset_->putAndInsertString(DCM_StudyInstanceUID, "1.2.3.4.5.6.7");
        test_dataset_->putAndInsertString(DCM_SeriesInstanceUID, "1.2.3.4.5.6");
        test_dataset_->putAndInsertString(DCM_Modality, "ECG");
        test_dataset_->putAndInsertString(DCM_PatientID, "TEST_PATIENT_001");
        test_dataset_->putAndInsertString(DCM_PatientName, "Test^Patient");
        test_dataset_->putAndInsertString(DCM_StudyDate, "20240101");
        
        // Create waveform sequence
        DcmSequenceOfItems* waveform_sequence = new DcmSequenceOfItems(DCM_WaveformSequence);
        DcmItem* waveform_item = new DcmItem();
        
        // Add waveform metadata
        waveform_item->putAndInsertUint16(DCM_NumberOfWaveformChannels, 3);
        waveform_item->putAndInsertUint32(DCM_NumberOfWaveformSamples, 1000);
        waveform_item->putAndInsertFloat64(DCM_SamplingFrequency, 250.0);
        waveform_item->putAndInsertUint16(DCM_WaveformBitsAllocated, 16);
        waveform_item->putAndInsertUint16(DCM_WaveformBitsStored, 12);
        
        // Add channel sensitivity and baseline
        DcmSequenceOfItems* sensitivity_sequence = new DcmSequenceOfItems(DCM_ChannelSensitivity);
        DcmSequenceOfItems* baseline_sequence = new DcmSequenceOfItems(DCM_ChannelBaseline);
        DcmSequenceOfItems* label_sequence = new DcmSequenceOfItems(DCM_ChannelLabel);
        
        for (int i = 0; i < 3; ++i) {
            DcmItem* sens_item = new DcmItem();
            sens_item->putAndInsertFloat64(DCM_ChannelSensitivity, 0.001); // 1mV per ADC unit
            sensitivity_sequence->insert(sens_item);
            
            DcmItem* base_item = new DcmItem();
            base_item->putAndInsertFloat64(DCM_ChannelBaseline, 0.0);
            baseline_sequence->insert(base_item);
            
            DcmItem* label_item = new DcmItem();
            label_item->putAndInsertString(DCM_ChannelLabel, ("Channel_" + std::to_string(i + 1)).c_str());
            label_sequence->insert(label_item);
        }
        
        waveform_item->insert(sensitivity_sequence);
        waveform_item->insert(baseline_sequence);
        waveform_item->insert(label_sequence);
        
        // Create synthetic waveform data (3 channels, 1000 samples each)
        std::vector<Uint16> waveform_data;
        waveform_data.reserve(3000); // 3 channels * 1000 samples
        
        // Generate synthetic ECG-like data
        for (int sample = 0; sample < 1000; ++sample) {
            for (int channel = 0; channel < 3; ++channel) {
                // Create a simple sine wave with different phases for each channel
                double value = 2048 + 500 * sin(2 * M_PI * sample / 250.0 + channel * M_PI / 3);
                waveform_data.push_back(static_cast<Uint16>(value));
            }
        }
        
        // Add waveform data
        waveform_item->putAndInsertUint16Array(DCM_WaveformData, waveform_data.data(), waveform_data.size());
        
        // Add item to sequence
        waveform_sequence->insert(waveform_item);
        
        // Add sequence to dataset
        test_dataset_->insert(waveform_sequence);
    }
    
    void TearDown() override {
        test_dataset_.reset();
    }
    
    std::unique_ptr<DcmDataset> test_dataset_;
};

TEST_F(WaveformExtractorTest, ExtractWaveformSuccess) {
    // Test successful waveform extraction
    auto result = extract_ecg_waveform(test_dataset_.get());
    
    ASSERT_TRUE(result.has_value()) << "Waveform extraction should succeed";
    
    const auto& waveform = result.value();
    
    // Verify basic properties
    EXPECT_EQ(waveform.num_channels(), 3);
    EXPECT_EQ(waveform.num_samples(), 1000);
    EXPECT_DOUBLE_EQ(waveform.sampling_frequency(), 250.0);
    EXPECT_EQ(waveform.samples_size(), 3000); // 3 channels * 1000 samples
    EXPECT_EQ(waveform.channel_names_size(), 3);
    EXPECT_DOUBLE_EQ(waveform.sensitivity(), 0.001);
    EXPECT_DOUBLE_EQ(waveform.baseline(), 0.0);
    EXPECT_EQ(waveform.units(), "mV");
    EXPECT_EQ(waveform.bits_allocated(), 16);
    EXPECT_EQ(waveform.bits_stored(), 12);
    EXPECT_TRUE(waveform.is_signed());
    
    // Verify channel names
    EXPECT_EQ(waveform.channel_names(0), "Channel_1");
    EXPECT_EQ(waveform.channel_names(1), "Channel_2");
    EXPECT_EQ(waveform.channel_names(2), "Channel_3");
    
    // Verify data range (should be reasonable for ECG)
    float min_val = *std::min_element(waveform.samples().begin(), waveform.samples().end());
    float max_val = *std::max_element(waveform.samples().begin(), waveform.samples().end());
    
    EXPECT_GT(min_val, -5.0f) << "Minimum value should be > -5mV";
    EXPECT_LT(max_val, 5.0f) << "Maximum value should be < 5mV";
}

TEST_F(WaveformExtractorTest, ExtractWaveformNoWaveformSequence) {
    // Create dataset without waveform sequence
    DcmDataset dataset;
    dataset.putAndInsertString(DCM_Modality, "ECG");
    
    auto result = extract_ecg_waveform(&dataset);
    
    EXPECT_FALSE(result.has_value()) << "Should return nullopt when no waveform sequence";
}

TEST_F(WaveformExtractorTest, ExtractWaveformEmptySequence) {
    // Create dataset with empty waveform sequence
    DcmDataset dataset;
    DcmSequenceOfItems* empty_sequence = new DcmSequenceOfItems(DCM_WaveformSequence);
    dataset.insert(empty_sequence);
    
    auto result = extract_ecg_waveform(&dataset);
    
    EXPECT_FALSE(result.has_value()) << "Should return nullopt when waveform sequence is empty";
}

TEST_F(WaveformExtractorTest, ValidateWaveformMedicalAccuracy) {
    // Test medical accuracy validation
    auto result = extract_ecg_waveform(test_dataset_.get());
    
    ASSERT_TRUE(result.has_value());
    
    const auto& waveform = result.value();
    
    // Should pass medical accuracy validation
    EXPECT_TRUE(validate_waveform_medical_accuracy(waveform));
    
    // Test with invalid parameters
    dicom::v1::WaveformData invalid_waveform;
    invalid_waveform.set_num_channels(0); // Invalid
    invalid_waveform.set_num_samples(1000);
    invalid_waveform.set_sampling_frequency(250.0);
    
    EXPECT_FALSE(validate_waveform_medical_accuracy(invalid_waveform));
    
    // Test with invalid sampling frequency
    invalid_waveform.set_num_channels(3);
    invalid_waveform.set_sampling_frequency(50.0); // Too low
    
    EXPECT_FALSE(validate_waveform_medical_accuracy(invalid_waveform));
    
    // Test with extreme values
    invalid_waveform.set_sampling_frequency(250.0);
    for (int i = 0; i < 100; ++i) {
        invalid_waveform.add_samples(100.0f); // Too high
    }
    
    EXPECT_FALSE(validate_waveform_medical_accuracy(invalid_waveform));
}

TEST_F(WaveformExtractorTest, ApplyDicomCalibration) {
    // Test DICOM calibration function
    uint16_t raw_value = 2048; // Middle of ADC range
    double sensitivity = 0.001; // 1mV per ADC unit
    double baseline = 0.0;
    
    float calibrated = apply_dicom_calibration(raw_value, sensitivity, baseline);
    
    EXPECT_FLOAT_EQ(calibrated, 2.048f) << "Calibrated value should be (2048 * 0.001) + 0.0";
    
    // Test with baseline
    baseline = 0.5;
    calibrated = apply_dicom_calibration(raw_value, sensitivity, baseline);
    
    EXPECT_FLOAT_EQ(calibrated, 2.548f) << "Calibrated value should be (2048 * 0.001) + 0.5";
    
    // Test with zero sensitivity (should prevent division by zero)
    sensitivity = 0.0;
    calibrated = apply_dicom_calibration(raw_value, sensitivity, baseline);
    
    EXPECT_FLOAT_EQ(calibrated, 2048.0f) << "Should use raw value when sensitivity is zero";
    
    // Test clamping
    sensitivity = 100.0; // Would produce very high value
    calibrated = apply_dicom_calibration(raw_value, sensitivity, baseline);
    
    EXPECT_LE(calibrated, 50.0f) << "Should clamp to maximum value";
}

TEST_F(WaveformExtractorTest, ExtractChannelMetadata) {
    // Test channel metadata extraction
    DcmItem item;
    
    // Add channel metadata
    item.putAndInsertString(DCM_ChannelLabel, "Test_Channel");
    item.putAndInsertFloat64(DCM_ChannelSensitivity, 0.002);
    item.putAndInsertFloat64(DCM_ChannelBaseline, 0.1);
    
    std::string channel_name;
    double sensitivity, baseline;
    
    bool result = extract_channel_metadata(&item, channel_name, sensitivity, baseline);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(channel_name, "Test_Channel");
    EXPECT_DOUBLE_EQ(sensitivity, 0.002);
    EXPECT_DOUBLE_EQ(baseline, 0.1);
    
    // Test with missing metadata
    DcmItem empty_item;
    result = extract_channel_metadata(&empty_item, channel_name, sensitivity, baseline);
    
    EXPECT_TRUE(result); // Should succeed with defaults
    EXPECT_EQ(channel_name, "");
    EXPECT_DOUBLE_EQ(sensitivity, 1.0);
    EXPECT_DOUBLE_EQ(baseline, 0.0);
}

TEST_F(WaveformExtractorTest, DeinterleaveWaveform) {
    // Test waveform deinterleaving
    std::vector<uint16_t> interleaved_data;
    
    // Create interleaved data: [ch1_s1, ch2_s1, ch3_s1, ch1_s2, ch2_s2, ch3_s2, ...]
    for (int sample = 0; sample < 100; ++sample) {
        for (int channel = 0; channel < 3; ++channel) {
            interleaved_data.push_back(channel * 100 + sample); // Unique value per channel and sample
        }
    }
    
    auto channel_data = deinterleave_waveform(interleaved_data, 3, 100);
    
    EXPECT_EQ(channel_data.size(), 3);
    EXPECT_EQ(channel_data[0].size(), 100);
    EXPECT_EQ(channel_data[1].size(), 100);
    EXPECT_EQ(channel_data[2].size(), 100);
    
    // Verify deinterleaving correctness
    for (int sample = 0; sample < 100; ++sample) {
        EXPECT_FLOAT_EQ(channel_data[0][sample], 0 * 100 + sample);
        EXPECT_FLOAT_EQ(channel_data[1][sample], 1 * 100 + sample);
        EXPECT_FLOAT_EQ(channel_data[2][sample], 2 * 100 + sample);
    }
}

TEST_F(WaveformExtractorTest, PerformanceTest) {
    // Test performance with large waveform
    // Create large waveform dataset
    DcmDataset large_dataset;
    
    // Add basic tags
    large_dataset.putAndInsertString(DCM_Modality, "ECG");
    
    // Create large waveform sequence (12 channels, 10000 samples)
    DcmSequenceOfItems* waveform_sequence = new DcmSequenceOfItems(DCM_WaveformSequence);
    DcmItem* waveform_item = new DcmItem();
    
    waveform_item->putAndInsertUint16(DCM_NumberOfWaveformChannels, 12);
    waveform_item->putAndInsertUint32(DCM_NumberOfWaveformSamples, 10000);
    waveform_item->putAndInsertFloat64(DCM_SamplingFrequency, 250.0);
    
    // Create large waveform data
    std::vector<Uint16> large_waveform_data;
    large_waveform_data.reserve(120000); // 12 channels * 10000 samples
    
    for (int sample = 0; sample < 10000; ++sample) {
        for (int channel = 0; channel < 12; ++channel) {
            large_waveform_data.push_back(static_cast<Uint16>(sample % 4096));
        }
    }
    
    waveform_item->putAndInsertUint16Array(DCM_WaveformData, large_waveform_data.data(), large_waveform_data.size());
    waveform_sequence->insert(waveform_item);
    large_dataset.insert(waveform_sequence);
    
    // Measure extraction time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = extract_ecg_waveform(&large_dataset);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    ASSERT_TRUE(result.has_value());
    
    const auto& waveform = result.value();
    EXPECT_EQ(waveform.num_channels(), 12);
    EXPECT_EQ(waveform.num_samples(), 10000);
    EXPECT_EQ(waveform.samples_size(), 120000);
    
    // Performance requirement: should complete within 50ms
    EXPECT_LT(duration.count(), 50) << "Large waveform extraction should complete within 50ms, took " << duration.count() << "ms";
}

// Integration test with real DICOM file (if available)
TEST(WaveformExtractorIntegrationTest, RealDICOMFile) {
    // This test requires a real DICOM file with waveform data
    // For now, we'll skip it if the file doesn't exist
    
    const std::string test_file_path = "test_data/sample_ecg_with_waveform.dcm";
    
    if (!std::filesystem::exists(test_file_path)) {
        GTEST_SKIP() << "Test DICOM file not found: " << test_file_path;
    }
    
    DcmFileFormat fileformat;
    OFCondition status = fileformat.loadFile(test_file_path.c_str());
    
    ASSERT_TRUE(status.good()) << "Failed to load test DICOM file: " << status.text();
    
    DcmDataset* dataset = fileformat.getDataset();
    ASSERT_NE(dataset, nullptr);
    
    auto result = extract_ecg_waveform(dataset);
    
    if (result.has_value()) {
        const auto& waveform = result.value();
        
        // Basic validation
        EXPECT_GT(waveform.num_channels(), 0);
        EXPECT_GT(waveform.num_samples(), 0);
        EXPECT_GT(waveform.sampling_frequency(), 0.0);
        EXPECT_GT(waveform.samples_size(), 0);
        
        // Medical accuracy
        EXPECT_TRUE(validate_waveform_medical_accuracy(waveform));
        
        std::cout << "Real DICOM waveform extracted: " 
                  << waveform.num_channels() << " channels, " 
                  << waveform.num_samples() << " samples, "
                  << waveform.sampling_frequency() << " Hz" << std::endl;
    } else {
        std::cout << "No waveform found in real DICOM file" << std::endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
