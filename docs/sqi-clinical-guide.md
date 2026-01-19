# Signal Quality Index (SQI) - Clinical Interpretation Guide

## Overview

This guide provides clinical interpretation of Signal Quality Index (SQI) values calculated using the Zhao & Zhang (2018) method. It helps healthcare professionals understand ECG signal quality and make informed decisions about diagnosis and treatment.

## Quality Levels

### Excellent Quality (SQI > 0.7)
**Clinical Significance**: Signal is suitable for reliable diagnosis and automated analysis.

**Characteristics**:
- Clear QRS complexes with consistent morphology
- Minimal baseline wander (<0.1mV)
- Low noise floor (<5% of signal power)
- Stable heart rate detection
- High signal-to-noise ratio (>10dB)

**Clinical Applications**:
- Automated arrhythmia detection
- Precise QT interval measurement
- Reliable ST-segment analysis
- AI-based diagnostic algorithms

**Action Required**: None - proceed with normal analysis

---

### Barely Acceptable Quality (0.4 ≤ SQI ≤ 0.7)
**Clinical Significance**: Signal has limitations but may be usable with caution.

**Characteristics**:
- Moderate noise level (5-15% of signal power)
- Some baseline wander present (0.1-0.3mV)
- Occasional QRS morphology variations
- Reduced signal-to-noise ratio (5-10dB)

**Clinical Applications**:
- Basic heart rate monitoring
- General rhythm assessment
- Preliminary screening (with confirmation)

**Action Required**: 
- Verify with additional measurements
- Consider signal filtering
- Use expert review for critical findings

---

### Unacceptable Quality (SQI < 0.4)
**Clinical Significance**: Signal is not suitable for clinical interpretation.

**Characteristics**:
- High noise levels (>15% of signal power)
- Significant baseline wander (>0.3mV)
- Poor QRS detection
- Low signal-to-noise ratio (<5dB)
- Frequent artifacts or interference

**Clinical Applications**:
- None suitable for diagnosis
- May indicate sensor disconnect
- Requires immediate attention

**Action Required**:
- Check electrode placement
- Verify sensor connections
- Replace sensors if necessary
- Reacquire signal after addressing issues

---

## Metric Interpretation

### pSQI (Power Spectrum SQI)
**Range**: 0.0 - 1.0

**Clinical Meaning**:
- **High (>0.7)**: Clear frequency distribution, QRS energy concentrated in 5-15Hz band
- **Medium (0.4-0.7)**: Some frequency contamination, moderate QRS energy
- **Low (<0.4)**: Significant frequency noise, poor QRS energy concentration

**Interpretation Guide**:
```
pSQI > 0.8: Excellent frequency characteristics
0.6 < pSQI ≤ 0.8: Good frequency characteristics
0.4 < pSQI ≤ 0.6: Fair frequency characteristics
pSQI ≤ 0.4: Poor frequency characteristics
```

---

### kSQI (Kurtosis SQI)
**Range**: 0.0 - 1.0

**Clinical Meaning**:
- **High (>0.8)**: Sharp, well-defined QRS complexes
- **Medium (0.4-0.8)**: Moderate QRS definition
- **Low (<0.4)**: Flattened or poorly defined QRS complexes

**Interpretation Guide**:
```
kSQI > 0.8: Excellent QRS morphology
0.6 < kSQI ≤ 0.8: Good QRS morphology
0.4 < kSQI ≤ 0.6: Fair QRS morphology
kSQI ≤ 0.4: Poor QRS morphology
```

---

### basSQI (Baseline SQI)
**Range**: 0.0 - 1.0

**Clinical Meaning**:
- **High (>0.7)**: Minimal baseline wander, stable isoelectric line
- **Medium (0.4-0.7)**: Moderate baseline wander
- **Low (<0.4)**: Significant baseline drift

**Interpretation Guide**:
```
basSQI > 0.8: Excellent baseline stability
0.6 < basSQI ≤ 0.8: Good baseline stability
0.4 < basSQI ≤ 0.6: Fair baseline stability
basSQI ≤ 0.4: Poor baseline stability
```

---

### qSQI (QRS Match SQI)
**Range**: 0.0 - 1.0

**Clinical Meaning**:
- **High (>0.8)**: Consistent RR intervals, physiological plausibility
- **Medium (0.4-0.8)**: Some RR irregularities
- **Low (<0.4)**: Significant RR interval irregularities

**Interpretation Guide**:
```
qSQI > 0.8: Excellent RR regularity
0.6 < qSQI ≤ 0.8: Good RR regularity
0.4 < qSQI ≤ 0.6: Fair RR regularity
qSQI ≤ 0.4: Poor RR regularity
```

---

### rSQI (R-peak SQI)
**Range**: 0.0 - 1.0

**Clinical Meaning**:
- **High (>0.7)**: Consistent R-peak amplitudes
- **Medium (0.4-0.7)**: Moderate amplitude variation
- **Low (<0.4)**: Significant amplitude inconsistency

**Interpretation Guide**:
```
rSQI > 0.7: Excellent amplitude consistency
0.6 < rSQI ≤ 0.7: Good amplitude consistency
0.4 < rSQI ≤ 0.6: Fair amplitude consistency
rSQI ≤ 0.4: Poor amplitude consistency
```

---

### sSQI (Spectral SQI)
**Range**: 0.0 - 1.0

**Clinical Meaning**:
- **High (>0.7)**: Pure frequency spectrum, minimal harmonics
- **Medium (0.4-0.7)**: Some spectral contamination
- **Low (<0.4)**: Significant spectral noise or harmonics

**Interpretation Guide**:
```
sSQI > 0.8: Excellent spectral purity
0.6 < sSQI ≤ 0.8: Good spectral purity
0.4 < sSQI ≤ 0.6: Fair spectral purity
sSQI ≤ 0.4: Poor spectral purity
```

---

## Clinical Decision Tree

### Signal Quality Assessment Flow

```
Start
│
├── Overall SQI ≥ 0.7?
│   ├── Yes → Excellent Quality
│   │   ├── Proceed with automated analysis
│   │   ├── Suitable for AI diagnostics
│   │   └── No intervention needed
│   │
│   └── No → Continue to next level
│
├── Overall SQI ≥ 0.4?
│   ├── Yes → Barely Acceptable Quality
│   │   ├── Check individual metrics
│   │   ├── Identify limiting factors
│   │   ├── Apply signal enhancement
│   │   └── Expert review recommended
│   │
│   └── No → Unacceptable Quality
│       ├── Check sensor connections
│       ├── Verify electrode placement
│       ├── Replace if necessary
│       └── Reacquire signal
```

---

## Integration with Clinical Workflow

### ECG Analysis Pipeline

1. **Signal Acquisition**
   - Verify sensor connections
   - Check electrode impedance
   - Ensure proper grounding

2. **Quality Assessment**
   - Calculate SQI using automated system
   - Review individual metrics
   - Identify quality limitations

3. **Quality-Based Routing**
   ```
   SQI ≥ 0.7 → Full Analysis Pipeline
   0.4 ≤ SQI < 0.7 → Enhanced Analysis + Review
   SQI < 0.4 → Signal Improvement Required
   ```

4. **Clinical Interpretation**
   - Consider quality limitations in diagnosis
   - Document quality-related uncertainties
   - Recommend repeat acquisition if needed

---

## Quality Improvement Guidelines

### Signal Enhancement Techniques

#### For Barely Acceptable Quality (0.4-0.7)

**1. Noise Reduction**
- Apply adaptive filtering
- Use wavelet denoising
- Implement baseline correction

**2. QRS Enhancement**
- Apply matched filtering
- Use template-based QRS detection
- Implement morphological smoothing

**3. Artifact Removal**
- Detect and remove power line interference
- Eliminate muscle noise artifacts
- Correct baseline wander

#### For Unacceptable Quality (<0.4)

**1. Immediate Actions**
- Check all sensor connections
- Verify electrode contact quality
- Ensure proper grounding

**2. Signal Reacquisition**
- Replace disposable electrodes
- Reposition sensors
- Check environmental interference

**3. Equipment Verification**
- Test amplifier performance
- Verify analog-to-digital conversion
- Check sampling rate stability

---

## Reporting Standards

### Quality Documentation Requirements

When reporting ECG analysis results, always include:

1. **SQI Score**: Overall quality score (0.0-1.0)
2. **Quality Level**: Classification (excellent/barely acceptable/unacceptable)
3. **Individual Metrics**: pSQI, kSQI, basSQI, qSQI, rSQI, sSQI
4. **Confidence Score**: Classification confidence (0.0-1.0)
5. **Processing Time**: Calculation duration in milliseconds
6. **Timestamp**: Analysis timestamp
7. **Recommendations**: Quality-based actions

### Example Report Format

```
ECG Quality Analysis Report
========================

Patient ID: PATIENT_001
Analysis Date: 2024-01-15 14:30:00 UTC
Signal Duration: 10 seconds
Sampling Rate: 360 Hz

Quality Assessment:
- Overall SQI: 0.73
- Quality Level: EXCELLENT
- Confidence: 0.85

Individual Metrics:
- pSQI (Power Spectrum): 0.82
- kSQI (Kurtosis): 0.78
- basSQI (Baseline): 0.71
- qSQI (QRS Match): 0.89
- rSQI (R-peak): 0.76
- sSQI (Spectral): 0.69

Interpretation:
Signal quality is excellent with well-defined QRS complexes,
minimal baseline wander, and low noise levels.
Suitable for automated arrhythmia detection and precise interval measurements.

Recommendations:
- Proceed with standard ECG analysis pipeline
- No signal enhancement required
- Results suitable for clinical interpretation

Processing Information:
- Calculation Time: 45ms
- System Performance: Within target (<100ms)
```

---

## Limitations and Considerations

### Technical Limitations

1. **Signal Length Dependency**: Requires minimum 10 seconds for reliable assessment
2. **Stationarity Assumption**: Assumes relative stationarity within segments
3. **Frequency Range**: Optimized for 0.5-40Hz ECG signals
4. **Artifact Sensitivity**: May be affected by transient artifacts

### Clinical Considerations

1. **Patient-Specific Factors**: Individual ECG morphology variations
2. **Acquisition Conditions**: Environmental interference impact
3. **Medical Context**: Consider patient condition and medications
4. **Regulatory Compliance**: Follow local medical device standards

---

## References

1. **Zhao, Z., & Zhang, Y. (2018)**. SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation. *Frontiers in Physiology*, 9, 727.

2. **Physionet Database**: https://physionet.org/content/sqidb/1.0.0/

3. **Clinical ECG Standards**: AHA/ACC/HRS recommendations for ECG quality

4. **IEC 60601-2-47**: Medical electrical equipment standards

---

## Support

For questions about SQI interpretation or implementation:
- Technical Support: development@vitalstream.com
- Clinical Consultation: medical@vitalstream.com
- Documentation: https://docs.vitalstream.com/sqi

*Last Updated: January 2024*
*Version: 1.0*
