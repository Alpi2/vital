#pragma once
#include <vector>

namespace VitalStream {

enum class AnomalyType {
    NONE = 0,
    TACHYCARDIA,
    BRADYCARDIA,
    PVC,
    AFIB,
    NOISE_ARTIFACT
};

class ECGGenerator {
public:
    ECGGenerator(int sample_rate = 360, float bpm = 72.0f);

    // EKG sinyali üretimi
    std::vector<float> generate_samples(int num_samples);

    // Parametre ayarları
    void set_bpm(float bpm);
    void set_noise_level(float level);
    void inject_anomaly(AnomalyType type);

    // Getter'lar
    float get_current_bpm() const;
    bool has_anomaly() const;

private:
    int sample_rate_;
    float bpm_;
    float noise_level_;
    double time_index_;
    AnomalyType anomaly_;

    // Dalga formülleri
    float p_wave(double t) const;
    float qrs_complex(double t) const;
    float t_wave(double t) const;
    float baseline_wander(double t) const;
};

} // namespace VitalStream
