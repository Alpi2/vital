#include "../include/ecg_generator.hpp"
#include <cmath>
#include <random>

namespace VitalStream {

ECGGenerator::ECGGenerator(int sample_rate, float bpm)
    : sample_rate_(sample_rate), bpm_(bpm), noise_level_(0.01f), time_index_(0.0), anomaly_(AnomalyType::NONE) {}

std::vector<float> ECGGenerator::generate_samples(int num_samples) {
    std::vector<float> out;
    out.reserve(num_samples);

    static thread_local std::mt19937 rng((std::random_device())());
    std::normal_distribution<float> noise_dist(0.0f, noise_level_);

    const double sample_dt = 1.0 / static_cast<double>(sample_rate_);
    const double beat_period = 60.0 / static_cast<double>(bpm_);

    for (int i = 0; i < num_samples; ++i) {
        double t = time_index_ * sample_dt; // seconds

        float value = 0.0f;
        // sum waveform components
        value += p_wave(t);
        value += qrs_complex(t);
        value += t_wave(t);
        value += baseline_wander(t);

        // simple anomaly effects
        if (anomaly_ == AnomalyType::PVC && fmod(t, beat_period) < 0.02)
            value += 0.6f; // a PVC spike

        // add gaussian noise
        value += noise_dist(rng);

        out.push_back(value);
        time_index_ += 1.0;
    }

    return out;
}

void ECGGenerator::set_bpm(float bpm) { bpm_ = bpm; }
void ECGGenerator::set_noise_level(float level) { noise_level_ = level; }
void ECGGenerator::inject_anomaly(AnomalyType type) { anomaly_ = type; }
float ECGGenerator::get_current_bpm() const { return bpm_; }
bool ECGGenerator::has_anomaly() const { return anomaly_ != AnomalyType::NONE; }

// Helper waveform implementations (simple gaussian-based pulses)
float ECGGenerator::p_wave(double t) const {
    const double period = 60.0 / static_cast<double>(bpm_);
    double phase = fmod(t, period);
    double center = 0.12 * period;
    double sigma = 0.02;
    double x = (phase - center) / sigma;
    return static_cast<float>(0.08 * std::exp(-0.5 * x * x));
}

float ECGGenerator::qrs_complex(double t) const {
    const double period = 60.0 / static_cast<double>(bpm_);
    double phase = fmod(t, period);
    double center = 0.2 * period;
    double sigma = 0.01;
    double x = (phase - center) / sigma;
    // sharper, higher amplitude
    return static_cast<float>(0.9 * std::exp(-0.5 * x * x));
}

float ECGGenerator::t_wave(double t) const {
    const double period = 60.0 / static_cast<double>(bpm_);
    double phase = fmod(t, period);
    double center = 0.36 * period;
    double sigma = 0.03;
    double x = (phase - center) / sigma;
    return static_cast<float>(0.18 * std::exp(-0.5 * x * x));
}

float ECGGenerator::baseline_wander(double t) const {
    // low-frequency baseline wander
    double freq = 0.33; // Hz
    return static_cast<float>(0.02 * std::sin(2.0 * M_PI * freq * t));
}

} // namespace VitalStream
