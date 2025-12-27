#pragma once
#include <vector>
#include "ecg_generator.hpp"

namespace VitalStream {

class ECG_Analyzer {
public:
    struct AnalysisResult {
        float heart_rate = 0.0f;
        float hr_variance = 0.0f;
        std::vector<int> r_peaks;
        AnomalyType detected_anomaly = AnomalyType::NONE;
        float confidence = 0.0f;
        int qrs_width = 0; // samples
    };

    ECG_Analyzer(int sample_rate = 360);

    // Ana analiz fonksiyonu
    AnalysisResult analyze(const std::vector<float>& signal);

    // Pan-Tompkins algoritması (basitleştirilmiş)
    std::vector<int> detect_r_peaks(const std::vector<float>& signal);

    // Anomali tespiti
    AnomalyType classify_anomaly(const AnalysisResult& result);

private:
    int sample_rate_;

    // Signal processing fonksiyonları
    std::vector<float> bandpass_filter(const std::vector<float>& signal);
    std::vector<float> derivative_filter(const std::vector<float>& signal);
    std::vector<float> squaring(const std::vector<float>& signal);
    std::vector<float> moving_window_integration(const std::vector<float>& signal);
};

} // namespace VitalStream
