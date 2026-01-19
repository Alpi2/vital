# VitalStream HL7 Integration Service

Enterprise-grade HL7/FHIR/DICOM integration service for VitalStream medical device platform.

## Overview

This Java Spring Boot service provides robust integration with healthcare information systems using industry-standard protocols:

- **HL7 v2.x**: Traditional healthcare messaging
- **HL7 FHIR R4**: Modern RESTful healthcare API
- **DICOM**: Medical imaging and waveform storage

## Features

### HL7 v2.x Support
- Message parsing and validation
- ORU^R01 (Observation Result) message creation
- ADT (Admission/Discharge/Transfer) support
- Customizable message templates

### FHIR R4 Support
- Observation resource creation for vital signs
- Patient resource management
- JSON and XML encoding/decoding
- FHIR server integration

### DICOM Support
- ECG waveform storage
- DICOM object creation
- Query/Retrieve operations
- PACS integration ready

## Technology Stack

- **Java 17**
- **Spring Boot 3.2.1**
- **HAPI FHIR 6.10.0** - Industry-leading FHIR library
- **HAPI HL7 v2** - HL7 v2.x message processing
- **DCM4CHE 5.31.0** - DICOM toolkit
- **Maven** - Build tool

## Architecture

```
hl7-integration-service/
├── src/main/java/com/vitalstream/hl7/
│   ├── HL7IntegrationApplication.java
│   ├── controller/
│   │   ├── HL7Controller.java
│   │   ├── FHIRController.java
│   │   └── DICOMController.java
│   ├── service/
│   │   ├── HL7Service.java
│   │   ├── FHIRService.java
│   │   └── DICOMService.java
│   └── model/
├── src/main/resources/
│   └── application.yml
└── pom.xml
```

## API Endpoints

### HL7 Endpoints

#### Parse HL7 Message
```bash
POST /integration/api/hl7/parse
Content-Type: text/plain

MSH|^~\&|VitalStream|Hospital|HIS|Hospital|20260103120000||ORU^R01|12345|P|2.5
```

#### Create Observation Message
```bash
POST /integration/api/hl7/observation
?patientId=PAT001
&type=HR
&value=75
&unit=bpm
```

#### Validate Message
```bash
POST /integration/api/hl7/validate
Content-Type: text/plain

[HL7 message string]
```

### FHIR Endpoints

#### Create Vital Sign Observation
```bash
POST /integration/api/fhir/observation
?patientId=PAT001
&code=8867-4
&display=Heart rate
&value=75
&unit=bpm
```

#### Create Patient
```bash
POST /integration/api/fhir/patient
?patientId=PAT001
&familyName=Doe
&givenName=John
&birthDate=1990-01-01
```

#### Submit Observation to FHIR Server
```bash
POST /integration/api/fhir/observation/submit
Content-Type: application/json

{
  "resourceType": "Observation",
  "status": "final",
  ...
}
```

## Configuration

Edit `src/main/resources/application.yml`:

```yaml
server:
  port: 8081

fhir:
  server:
    url: http://your-fhir-server:8080/fhir

logging:
  level:
    com.vitalstream: DEBUG
```

## Building and Running

### Prerequisites
- Java 17 or higher
- Maven 3.6+

### Build
```bash
cd hl7-integration-service
mvn clean package
```

### Run
```bash
java -jar target/hl7-integration-service-1.0.0.jar
```

Or with Maven:
```bash
mvn spring-boot:run
```

### Docker
```bash
docker build -t vitalstream-hl7-integration .
docker run -p 8081:8081 vitalstream-hl7-integration
```

## Integration with VitalStream

### From Rust Device Drivers
```rust
// Send HL7 message via HTTP
let client = reqwest::Client::new();
let response = client
    .post("http://localhost:8081/integration/api/hl7/observation")
    .query(&[
        ("patientId", "PAT001"),
        ("type", "HR"),
        ("value", "75"),
        ("unit", "bpm"),
    ])
    .send()
    .await?;
```

### From Python Backend
```python
import requests

response = requests.post(
    'http://localhost:8081/integration/api/fhir/observation',
    params={
        'patientId': 'PAT001',
        'code': '8867-4',
        'display': 'Heart rate',
        'value': 75,
        'unit': 'bpm'
    }
)
```

## Testing

### Unit Tests
```bash
mvn test
```

### Integration Tests
```bash
mvn verify
```

### Manual Testing with cURL

```bash
# Test HL7 observation creation
curl -X POST "http://localhost:8081/integration/api/hl7/observation?patientId=PAT001&type=HR&value=75&unit=bpm"

# Test FHIR observation creation
curl -X POST "http://localhost:8081/integration/api/fhir/observation?patientId=PAT001&code=8867-4&display=Heart%20rate&value=75&unit=bpm"
```

## Compliance

### FDA/IEC 62304
- Comprehensive error handling
- Audit logging
- Input validation
- Secure communication

### HL7 Standards
- HL7 v2.5 compliant
- FHIR R4 compliant
- DICOM 3.0 compliant

## Monitoring

Health check endpoint:
```bash
curl http://localhost:8081/integration/actuator/health
```

Metrics:
```bash
curl http://localhost:8081/integration/actuator/metrics
```

## Deployment

### Production Considerations
1. Configure HTTPS/TLS
2. Set up authentication (OAuth2/JWT)
3. Configure FHIR server URL
4. Set up log aggregation
5. Configure monitoring and alerting
6. Set up database for message persistence

### Environment Variables
```bash
export FHIR_SERVER_URL=https://fhir.hospital.com/fhir
export SERVER_PORT=8081
export SPRING_PROFILES_ACTIVE=production
```

## Troubleshooting

### Common Issues

1. **FHIR server connection failed**
   - Check `fhir.server.url` in application.yml
   - Verify network connectivity

2. **HL7 parsing errors**
   - Validate message format
   - Check field separators

3. **DICOM storage failed**
   - Verify file permissions
   - Check disk space

## License

MIT License

## Support

For issues and questions, please open a GitHub issue.
