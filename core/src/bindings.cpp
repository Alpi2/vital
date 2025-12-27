#include <emscripten/bind.h>
#include "ecg_generator.hpp"
#include "ecg_analyzer.hpp"

using namespace emscripten;
using namespace VitalStream;

EMSCRIPTEN_BINDINGS(vitalstream_module) {
    class_<ECGGenerator>("ECGGenerator")
        .constructor<int, float>()
        .function("generateSamples", &ECGGenerator::generate_samples)
        .function("setBPM", &ECGGenerator::set_bpm)
        .function("setNoiseLevel", &ECGGenerator::set_noise_level)
        .function("injectAnomaly", &ECGGenerator::inject_anomaly)
        .property("currentBPM", &ECGGenerator::get_current_bpm)
        .property("hasAnomaly", &ECGGenerator::has_anomaly);
    
    class_<ECG_Analyzer>("ECGAnalyzer")
        .constructor<int>()
        .function("analyze", &ECG_Analyzer::analyze);
    
    value_object<ECG_Analyzer::AnalysisResult>("AnalysisResult")
        .field("heartRate", &ECG_Analyzer::AnalysisResult::heart_rate)
        .field("hrVariance", &ECG_Analyzer::AnalysisResult::hr_variance)
        .field("detectedAnomaly", &ECG_Analyzer::AnalysisResult::detected_anomaly)
        .field("confidence", &ECG_Analyzer::AnalysisResult::confidence);
    
    enum_<AnomalyType>("AnomalyType")
        .value("NONE", AnomalyType::NONE)
        .value("TACHYCARDIA", AnomalyType::TACHYCARDIA)
        .value("BRADYCARDIA", AnomalyType::BRADYCARDIA)
        .value("PVC", AnomalyType::PVC)
        .value("AFIB", AnomalyType::AFIB)
        .value("NOISE_ARTIFACT", AnomalyType::NOISE_ARTIFACT);
}
