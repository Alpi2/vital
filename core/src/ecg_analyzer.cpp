#include "../include/ecg_analyzer.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace VitalStream {

ECG_Analyzer::ECG_Analyzer(int sample_rate) : sample_rate_(sample_rate) {}

static std::vector<float> moving_average(const std::vector<float>& x, int win) {
    std::vector<float> y(x.size(), 0.0f);
    if (win <= 1) return x;
    float inv = 1.0f / win;
    float sum = 0.0f;
    int half = win / 2;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i];
        if (i >= (size_t)win) sum -= x[i - win];
        if (i >= (size_t)win - 1) y[i - (win - 1) / 2] = sum * inv;
    }
    return y;
}

std::vector<float> ECG_Analyzer::bandpass_filter(const std::vector<float>& signal) {
    // Very simple bandpass: lowpass then highpass by subtraction of slow-moving average
    auto low = moving_average(signal, 5); // smooth high freq
    // create a slow baseline for highpass
    auto baseline = moving_average(signal, std::max(1, sample_rate_ / 2));
    std::vector<float> out(signal.size(), 0.0f);
    for (size_t i = 0; i < signal.size(); ++i) {
        float l = (i < low.size()) ? low[i] : signal[i];
        float b = (i < baseline.size()) ? baseline[i] : 0.0f;
        out[i] = l - b;
    }
    return out;
}

std::vector<float> ECG_Analyzer::derivative_filter(const std::vector<float>& signal) {
    std::vector<float> out(signal.size(), 0.0f);
    for (size_t i = 1; i < signal.size(); ++i) out[i] = signal[i] - signal[i - 1];
    return out;
}

std::vector<float> ECG_Analyzer::squaring(const std::vector<float>& signal) {
    std::vector<float> out(signal.size(), 0.0f);
    for (size_t i = 0; i < signal.size(); ++i) out[i] = signal[i] * signal[i];
    return out;
}

std::vector<float> ECG_Analyzer::moving_window_integration(const std::vector<float>& signal) {
    int win = std::max(1, static_cast<int>(0.15 * sample_rate_));
    std::vector<float> out(signal.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < signal.size(); ++i) {
        sum += signal[i];
        if (i >= (size_t)win) sum -= signal[i - win];
        out[i] = sum / win;
    }
    return out;
}

std::vector<int> ECG_Analyzer::detect_r_peaks(const std::vector<float>& signal) {
    // Pan-Tompkins simplified pipeline
    auto bp = bandpass_filter(signal);
    auto der = derivative_filter(bp);
    auto sq = squaring(der);
    auto mwi = moving_window_integration(sq);

    // thresholding
    float mean = 0.0f;
    for (auto v : mwi) mean += v;
    mean /= std::max<size_t>(1, mwi.size());
    float sqsum = 0.0f;
    for (auto v : mwi) sqsum += (v - mean) * (v - mean);
    float stddev = std::sqrt(sqsum / std::max<size_t>(1, mwi.size()));
    float thresh = mean + 0.5f * stddev;

    std::vector<int> peaks;
    int refractory = static_cast<int>(0.2f * sample_rate_); // 200ms
    int last_peak = -refractory;
    for (size_t i = 1; i + 1 < mwi.size(); ++i) {
        if (mwi[i] > thresh && mwi[i] > mwi[i - 1] && mwi[i] > mwi[i + 1]) {
            if (static_cast<int>(i) - last_peak > refractory) {
                peaks.push_back(static_cast<int>(i));
                last_peak = static_cast<int>(i);
            }
        }
    }
    return peaks;
}

AnomalyType ECG_Analyzer::classify_anomaly(const AnalysisResult& result) {
    if (result.heart_rate > 100.0f) return AnomalyType::TACHYCARDIA;
    if (result.heart_rate < 60.0f) return AnomalyType::BRADYCARDIA;
    return AnomalyType::NONE;
}

ECG_Analyzer::AnalysisResult ECG_Analyzer::analyze(const std::vector<float>& signal) {
    AnalysisResult res;
    if (signal.empty()) return res;

    res.r_peaks = detect_r_peaks(signal);
    if (res.r_peaks.size() >= 2) {
        std::vector<float> rr_intervals;
        for (size_t i = 1; i < res.r_peaks.size(); ++i) {
            float dt = static_cast<float>(res.r_peaks[i] - res.r_peaks[i - 1]) / static_cast<float>(sample_rate_);
            rr_intervals.push_back(dt);
        }
        float mean_rr = std::accumulate(rr_intervals.begin(), rr_intervals.end(), 0.0f) / rr_intervals.size();
        res.heart_rate = 60.0f / mean_rr;
        // variance
        float var = 0.0f;
        for (auto v : rr_intervals) var += (v - mean_rr) * (v - mean_rr);
        res.hr_variance = var / rr_intervals.size();
        // estimate QRS width roughly from original signal around first peak
        int peak = res.r_peaks.front();
        float peak_val = signal[peak];
        float half = peak_val * 0.5f;
        int left = peak, right = peak;
        while (left > 0 && signal[left] > half) --left;
        while (right + 1 < static_cast<int>(signal.size()) && signal[right] > half) ++right;
        res.qrs_width = right - left;
    }

    res.detected_anomaly = classify_anomaly(res);
    // confidence heuristic: more peaks and low variance -> higher confidence
    res.confidence = std::min(1.0f, 0.2f + 0.8f * (std::min<size_t>(res.r_peaks.size(), 10) / 10.0f));
    return res;
}

} // namespace VitalStream
