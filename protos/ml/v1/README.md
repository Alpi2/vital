# ML Inference Service Protocol Buffers

## Overview
The ML Inference Service provides machine learning capabilities for medical data analysis and predictions.

## Messages

### Core Messages
- **ModelInfo** - Model metadata and configuration
- **InputData** - Input data for inference
- **InferenceRequest** - Request for model inference
- **InferenceResponse** - Inference results with predictions
- **PredictionResult** - Individual prediction with confidence

### Model Management
- **TrainModelRequest/Response** - Train new models
- **EvaluateModelRequest/Response** - Evaluate model performance
- **DeployModelRequest/Response** - Deploy models to production
- **ModelRegistryEntry** - Model registry information

### Monitoring
- **ModelMonitoring** - Real-time model monitoring data
- **GetModelMonitoringRequest/Response** - Query monitoring metrics

### Batch Operations
- **BatchInferenceRequest/Response** - Execute multiple inferences
- **StreamInferenceRequest/Response** - Real-time inference streaming

## Enums

### ModelType
- `MODEL_TYPE_UNSPECIFIED` (0)
- `MODEL_TYPE_ECG_CLASSIFICATION` (1)
- `MODEL_TYPE_ARRHYTHMIA_DETECTION` (2)
- `MODEL_TYPE_HEART_RATE_VARIABILITY` (3)
- `MODEL_TYPE_ANOMALY_DETECTION` (4)
- `MODEL_TYPE_PREDICTIVE_ANALYTICS` (5)
- `MODEL_TYPE_RISK_ASSESSMENT` (6)
- `MODEL_TYPE_SEPSIS_PREDICTION` (7)
- `MODEL_TYPE_MORTALITY_PREDICTION` (8)
- `MODEL_TYPE_READMISSION_PREDICTION` (9)
- `MODEL_TYPE_DRUG_RESPONSE` (10)
- `MODEL_TYPE_IMAGE_CLASSIFICATION` (11)
- `MODEL_TYPE_NATURAL_LANGUAGE_PROCESSING` (12)
- `MODEL_TYPE_TIME_SERIES_FORECASTING` (13)
- `MODEL_TYPE_CLAIM_DETECTION` (14)
- `MODEL_TYPE_FRAUD_DETECTION` (15)

### InferenceStatus
- `INFERENCE_STATUS_UNSPECIFIED` (0)
- `INFERENCE_STATUS_PENDING` (1)
- `INFERENCE_STATUS_PROCESSING` (2)
- `INFERENCE_STATUS_COMPLETED` (3)
- `INFERENCE_STATUS_FAILED` (4)
- `INFERENCE_STATUS_CANCELLED` (5)
- `INFERENCE_STATUS_TIMEOUT` (6)

### ModelStatus
- `MODEL_STATUS_UNSPECIFIED` (0)
- `MODEL_STATUS_TRAINING` (1)
- `MODEL_STATUS_TRAINED` (2)
- `MODEL_STATUS_VALIDATING` (3)
- `MODEL_STATUS_DEPLOYED` (4)
- `MODEL_STATUS_DEPRECATED` (5)
- `MODEL_STATUS_ERROR` (6)
- `MODEL_STATUS_MAINTENANCE` (7)

### InputDataType
- `INPUT_DATA_TYPE_UNSPECIFIED` (0)
- `INPUT_DATA_TYPE_ECG_SIGNAL` (1)
- `INPUT_DATA_TYPE_PPG_SIGNAL` (2)
- `INPUT_DATA_TYPE_BLOOD_PRESSURE` (3)
- `INPUT_DATA_TYPE_GLUCOSE_LEVEL` (4)
- `INPUT_DATA_TYPE_TEMPERATURE` (5)
- `INPUT_DATA_TYPE_SPO2` (6)
- `INPUT_DATA_TYPE_RESPIRATION_RATE` (7)
- `INPUT_DATA_TYPE_ACTIVITY_DATA` (8)
- `INPUT_DATA_TYPE_SLEEP_DATA` (9)
- `INPUT_DATA_TYPE_MEDICATION_DATA` (10)
- `INPUT_DATA_TYPE_LAB_RESULTS` (11)
- `INPUT_DATA_TYPE_VITAL_SIGNS` (12)
- `INPUT_DATA_TYPE_CLINICAL_NOTES` (13)
- `INPUT_DATA_TYPE_MEDICAL_IMAGE` (14)
- `INPUT_DATA_TYPE_GENOMIC_DATA` (15)

## Service RPCs

### Core Inference
- `Inference` - Single inference request
- `BatchInference` - Multiple inference requests
- `StreamInference` - Real-time inference streaming

### Model Management
- `TrainModel` - Train new ML models
- `EvaluateModel` - Evaluate model performance
- `DeployModel` - Deploy models to production
- `GetModelRegistry` - Query model registry

### Monitoring
- `GetModelMonitoring` - Get model monitoring data
- `ModelChannel` - Bidirectional model communication
- `HealthCheck` - Service health check

## Usage Examples

### Single Inference
```python
request = InferenceRequest(
    model_id="ecg-classifier-v1",
    inputs=[
        InputData(
            type=InputDataType.INPUT_DATA_TYPE_ECG_SIGNAL,
            data=ecg_signal_data,
            timestamp=datetime.now(),
            patient_id="patient-123"
        )
    ],
    include_confidence=True,
    include_explanations=True
)
```

### Batch Inference
```python
request = BatchInferenceRequest(
    requests=[
        InferenceRequest(model_id="model-1", inputs=inputs1),
        InferenceRequest(model_id="model-2", inputs=inputs2)
    ],
    continue_on_error=True
)
```

### Train Model
```python
request = TrainModelRequest(
    model_name="custom-ecg-model",
    type=ModelType.MODEL_TYPE_ECG_CLASSIFICATION,
    training_data=training_dataset,
    validation_data=validation_dataset,
    hyperparameters={"learning_rate": 0.001, "batch_size": 32}
)
```

### Stream Inference
```python
request = StreamInferenceRequest(
    model_id="real-time-arrhythmia-detector",
    input_type=InputDataType.INPUT_DATA_TYPE_ECG_SIGNAL,
    real_time=True,
    sampling_rate=250
)
```

## Best Practices

1. **Model Selection**
   - Choose appropriate model type for use case
   - Consider model performance metrics
   - Validate model before deployment

2. **Data Preparation**
   - Use correct input data types
   - Ensure data quality and format
   - Include metadata for traceability

3. **Inference Requests**
   - Set appropriate timeouts
   - Include confidence scores when needed
   - Handle async responses properly

4. **Model Training**
   - Use sufficient training data
   - Include validation datasets
   - Monitor training progress

5. **Performance**
   - Use batch processing for multiple requests
   - Implement streaming for real-time needs
   - Monitor model performance continuously

## Model Lifecycle

1. **Development** - Create and train models
2. **Validation** - Test model performance
3. **Deployment** - Deploy to production
4. **Monitoring** - Track performance and drift
5. **Retraining** - Update models with new data
6. **Deprecation** - Retire old models

## Error Handling

- Check inference status in responses
- Handle model loading errors gracefully
- Implement retry logic for transient failures
- Monitor model performance and accuracy
- Log all inference requests and results
