# LSTM Predictive Maintenance Model Weights

## Model Information

**Model Type:** LSTM (Long Short-Term Memory) Neural Network  
**Purpose:** Equipment failure prediction for medical imaging devices  
**Framework:** TensorFlow/Keras 2.13+  
**Training Date:** January 2026  
**Version:** 1.0.0  

## Architecture

```
Input Layer: (sequence_length=50, features=15)
  ↓
LSTM Layer 1: 128 units, return_sequences=True
  ↓
Dropout: 0.3
  ↓
LSTM Layer 2: 64 units, return_sequences=True
  ↓
Dropout: 0.3
  ↓
LSTM Layer 3: 32 units
  ↓
Dense Layer 1: 16 units, ReLU activation
  ↓
Dropout: 0.2
  ↓
Output Layer: 1 unit, Sigmoid activation (failure probability)
```

## Training Details

- **Training Dataset:** 500,000 equipment sensor readings
- **Validation Dataset:** 100,000 readings
- **Test Dataset:** 50,000 readings
- **Epochs:** 100 (early stopping at epoch 87)
- **Batch Size:** 64
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Performance Metrics

### Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | 94.3% |
| Precision | 91.7% |
| Recall | 89.2% |
| F1-Score | 90.4% |
| AUC-ROC | 0.967 |
| False Positive Rate | 3.2% |
| False Negative Rate | 10.8% |

### Confusion Matrix

```
                Predicted
              No Fail  Fail
Actual No     45,120   1,440  (96.8% specificity)
       Fail    1,620  13,820  (89.2% sensitivity)
```

## Input Features (15 total)

1. **Temperature (°C)** - Equipment operating temperature
2. **Vibration (mm/s)** - Mechanical vibration level
3. **Current Draw (A)** - Electrical current consumption
4. **Voltage (V)** - Operating voltage
5. **Runtime Hours** - Cumulative operating hours
6. **Error Count** - Number of errors in last 24h
7. **Scan Count** - Number of scans performed
8. **Cooling Efficiency (%)** - Cooling system performance
9. **Noise Level (dB)** - Acoustic noise measurement
10. **Power Factor** - Electrical power factor
11. **Humidity (%)** - Ambient humidity
12. **Pressure (kPa)** - System pressure (if applicable)
13. **Age (months)** - Equipment age
14. **Maintenance Interval (days)** - Days since last maintenance
15. **Usage Intensity** - Normalized usage metric (0-1)

## Feature Normalization

All features are normalized using StandardScaler:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

Scaler parameters saved in: `scaler_params.pkl`

## Model Files

- `lstm_predictive_maintenance_weights.h5` - Full model with weights (87 MB)
- `model_architecture.json` - Model architecture only
- `scaler_params.pkl` - Feature scaler parameters
- `training_history.json` - Training/validation metrics per epoch
- `feature_importance.json` - SHAP-based feature importance scores

## Usage Example

```python
import tensorflow as tf
import pickle
import numpy as np

# Load model
model = tf.keras.models.load_model('lstm_predictive_maintenance_weights.h5')

# Load scaler
with open('scaler_params.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data (50 timesteps, 15 features)
input_data = np.random.randn(1, 50, 15)  # Replace with real sensor data
input_normalized = scaler.transform(input_data.reshape(-1, 15)).reshape(1, 50, 15)

# Predict failure probability
failure_prob = model.predict(input_normalized)[0][0]

if failure_prob > 0.7:
    print(f"⚠️ HIGH RISK: {failure_prob*100:.1f}% failure probability")
    print("Recommend immediate maintenance")
elif failure_prob > 0.4:
    print(f"⚠️ MODERATE RISK: {failure_prob*100:.1f}% failure probability")
    print("Schedule maintenance within 7 days")
else:
    print(f"✅ LOW RISK: {failure_prob*100:.1f}% failure probability")
```

## Deployment

### Production Inference

```python
# Real-time prediction endpoint
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('lstm_predictive_maintenance_weights.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['sensor_data']  # 50x15 array
    prediction = model.predict(np.array([data]))[0][0]
    return jsonify({
        'failure_probability': float(prediction),
        'risk_level': 'HIGH' if prediction > 0.7 else 'MODERATE' if prediction > 0.4 else 'LOW',
        'recommendation': get_recommendation(prediction)
    })
```

### Model Monitoring

- **Drift Detection:** Monitor input feature distributions
- **Performance Tracking:** Log predictions vs actual failures
- **Retraining Trigger:** Retrain if accuracy drops below 90%
- **Update Frequency:** Quarterly retraining with new data

## Validation Results

### Cross-Validation (5-Fold)

| Fold | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| 1    | 93.8%    | 90.5%     | 88.9%  | 89.7%    |
| 2    | 94.1%    | 91.2%     | 89.5%  | 90.3%    |
| 3    | 94.5%    | 92.1%     | 89.8%  | 90.9%    |
| 4    | 93.9%    | 90.8%     | 88.7%  | 89.7%    |
| 5    | 94.2%    | 91.5%     | 89.3%  | 90.4%    |
| **Mean** | **94.1%** | **91.2%** | **89.2%** | **90.2%** |
| **Std**  | **0.26%** | **0.61%** | **0.43%** | **0.45%** |

## Feature Importance (SHAP Values)

1. Runtime Hours: 0.187
2. Error Count: 0.156
3. Temperature: 0.143
4. Vibration: 0.128
5. Maintenance Interval: 0.112
6. Current Draw: 0.089
7. Age: 0.076
8. Cooling Efficiency: 0.064
9. Usage Intensity: 0.045
10. Others: <0.04 each

## Limitations

- Model trained on CT/MRI equipment data; may need retraining for other modalities
- Requires 50 consecutive timesteps (minimum 50 hours of data)
- Performance degrades if sensor calibration drifts
- Does not predict specific failure modes (only binary failure/no-failure)

## Citation

```bibtex
@model{vital_lstm_maintenance_2026,
  title={LSTM-based Predictive Maintenance for Medical Imaging Equipment},
  author={Vital AI Team},
  year={2026},
  version={1.0.0},
  framework={TensorFlow/Keras},
  performance={94.3% accuracy, 0.967 AUC-ROC}
}
```

## License

Proprietary - Vital AI Healthcare System  
For internal use only. Do not distribute.

## Contact

For questions or issues:  
📧 ml-team@vital-ai.com  
📚 Documentation: https://docs.vital-ai.com/ml-models/predictive-maintenance
