# GDPR Implementation Guide

## Overview

This document provides comprehensive guidance for the GDPR compliance implementation in the VitalStream healthcare platform. The implementation follows all GDPR Articles and EDPB guidelines for processing health data of EU citizens.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Legal Compliance](#legal-compliance)
3. [Implementation Components](#implementation-components)
4. [Security Measures](#security-measures)
5. [Data Subject Rights](#data-subject-rights)
6. [Consent Management](#consent-management)
7. [Data Retention](#data-retention)
8. [Anonymization](#anonymization)
9. [Breach Management](#breach-management)
10. [Monitoring & Compliance](#monitoring--compliance)
11. [Testing](#testing)
12. [Deployment](#deployment)
13. [Maintenance](#maintenance)

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Background    │
│   (Angular)     │◄──►│   (FastAPI)     │◄──►│   Tasks (Celery) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Privacy UI    │    │   GDPR Services │    │   Compliance    │
│   Components    │    │   & Models      │    │   Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User          │    │   PostgreSQL    │    │   Redis Queue   │
│   Interface     │    │   Database      │    │   & Storage     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Services

1. **ConsentService**: Manages GDPR Article 7 consent requirements
2. **DataSubjectRightsService**: Implements Articles 15-22 data subject rights
3. **AnonymizationService**: Provides EDPB-compliant data anonymization
4. **RetentionService**: Handles purpose-based data retention by country
5. **MonitoringService**: Real-time compliance monitoring and alerting

## Legal Compliance

### GDPR Articles Implemented

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| **Article 6** | Lawful basis for processing | Legal basis enumeration and validation |
| **Article 7** | Conditions for consent | Explicit consent with audit trails |
| **Article 8** | Protection of children | Age verification (16+ for health data) |
| **Article 9** | Special category data | Explicit consent for health data |
| **Article 15** | Right of access | Automated data export (JSON/CSV/FHIR) |
| **Article 16** | Right to rectification | Data correction workflow |
| **Article 17** | Right to erasure | Automated deletion with legal exceptions |
| **Article 18** | Right to restriction | Processing restriction management |
| **Article 20** | Right to data portability | FHIR R4 export and direct transfer |
| **Article 21** | Right to object | Objection handling and opt-out |
| **Article 22** | Automated decision-making | AI decision transparency |
| **Article 25** | Data protection by design | Privacy by design principles |
| **Article 30** | Records of processing | ROPA documentation |
| **Article 32** | Security of processing | Encryption, access controls |
| **Article 33** | Breach notification | 72-hour authority notification |
| **Article 34** | Communication of breach | Subject notification for high risk |
| **Article 35** | DPIA | Automated impact assessments |
| **Article 37** | DPO designation | DPO contact and dashboard |

### Country-Specific Requirements

The implementation supports all EU member states with country-specific retention periods:

| Country | Medical Records | Research Data | Legal Basis |
|---------|----------------|--------------|-------------|
| **Germany** | 10 years | 15 years | MedDV § 10 |
| **France** | 20 years | 15 years | Public Health Code L.1111-7 |
| **UK** | 8 years | 20 years | DPA 2018 Schedule 1 |
| **Italy** | 10 years | 25 years | Privacy Code |
| **Spain** | 5 years | 15 years | Data Protection Act |
| **Netherlands** | 15 years | 15 years | Medical Treatment Contracts Act |

## Implementation Components

### Backend Services

#### 1. Consent Service (`app/services/consent_service.py`)

```python
class ConsentService:
    async def grant_consent(request: ConsentRequest) -> ConsentResponse
    async def revoke_consent(consent_id: str, patient_id: str, reason: str) -> ConsentResponse
    async def check_consent(patient_id: str, consent_type: ConsentType) -> Tuple[bool, Optional[PatientConsent]]
    async def bulk_update_consents(patient_id: str, consents: List[Dict]) -> ConsentResponse
    async def export_consent_proof(patient_id: str) -> ConsentProofResponse
    async def expire_old_consents() -> int
```

**Key Features:**
- Explicit consent with detailed audit trails
- IP address and user agent tracking
- Version control for consent changes
- Withdrawal effects and cascade handling
- Digital signature consent proof

#### 2. Data Subject Rights Service (`app/services/data_subject_rights_service.py`)

```python
class DataSubjectRightsService:
    async def create_access_request(request: AccessRequest) -> Dict
    async def create_erasure_request(request: ErasureRequest) -> Dict
    async def create_portability_request(request: PortabilityRequest) -> Dict
    async def process_access_request(request_id: str) -> Dict
    async def process_erasure_request(request_id: str) -> Dict
    async def initiate_step_up_verification(patient_id: str, method: str) -> Dict
```

**Key Features:**
- Step-up authentication for sensitive operations
- Multiple export formats (JSON, CSV, FHIR R4)
- Legal exception handling for erasure
- Automated processing within 30-day deadline
- Secure download mechanisms

#### 3. Anonymization Service (`app/services/anonymization_service.py`)

```python
class AnonymizationService:
    async def anonymize_patient_data(patient_id: str, config: AnonymizationConfig) -> AnonymizationResult
    async def check_anonymization_quality(table_name: str) -> Dict
    async def pseudonymize_data(data: Dict) -> Dict
    async def validate_anonymization(anonymized_data: List[Dict]) -> Dict
```

**Key Features:**
- k-anonymity (k≥5), l-diversity (l≥3), t-closeness validation
- Differential privacy with ε≤1.0
- Proven library integration (Presidio, pyanonymization)
- Quality assessment and risk evaluation

### Database Models

#### 1. Patient Consent (`app/models/gdpr.py`)

```python
class PatientConsent(Base):
    __tablename__ = "patient_consents"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    consent_type = Column(Enum(ConsentType))
    granted = Column(Boolean, default=False)
    granted_at = Column(DateTime(timezone=True))
    revoked_at = Column(DateTime(timezone=True))
    version = Column(Integer, default=1)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    consent_text = Column(Text)
    legal_basis = Column(Enum(LegalBasisType))
    purpose = Column(Text)
    expiry_date = Column(DateTime(timezone=True))
    withdrawal_effects = Column(JSON)
```

#### 2. Data Subject Request (`app/models/gdpr.py`)

```python
class DataSubjectRequest(Base):
    __tablename__ = "data_subject_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    request_type = Column(Enum(DataSubjectRequestType))
    status = Column(Enum(RequestStatus))
    request_date = Column(DateTime(timezone=True))
    deadline = Column(DateTime(timezone=True))
    format = Column(String(10))
    delivery_method = Column(String(20))
    download_url = Column(Text)
    download_expires = Column(DateTime(timezone=True))
    exceptions = Column(JSON)
```

### API Endpoints

#### Consent Management

```
POST   /api/v1/gdpr/consent                    # Grant consent
DELETE /api/v1/gdpr/consent/{consent_id}        # Revoke consent
GET    /api/v1/gdpr/consent                    # List consents
POST   /api/v1/gdpr/consent/bulk               # Bulk update consents
GET    /api/v1/gdpr/consent/{consent_id}/proof  # Get consent proof
```

#### Data Subject Rights

```
POST /api/v1/gdpr/access-request              # Create access request
POST /api/v1/gdpr/erasure-request             # Create erasure request
POST /api/v1/gdpr/portability-request         # Create portability request
GET  /api/v1/gdpr/requests/{request_id}       # Get request status
GET  /api/v1/gdpr/requests                    # List all requests
GET  /api/v1/gdpr/download/{request_id}       # Download data export
```

#### Step-up Authentication

```
POST /api/v1/gdpr/verify                       # Initiate verification
POST /api/v1/gdpr/verify/{verification_id}    # Complete verification
```

#### Documentation

```
GET /api/v1/gdpr/privacy-policy               # Get privacy policy
GET /api/v1/gdpr/dpia                         # Get DPIA document
GET /api/v1/gdpr/ropa                         # Get ROPA document
```

## Security Measures

### Authentication & Authorization

1. **JWT Tokens**: Secure authentication with expiration
2. **RBAC**: Role-based access control (patient, admin, dpo)
3. **Step-up Authentication**: Additional verification for sensitive operations
4. **Rate Limiting**: 10 requests/hour for sensitive endpoints

### Data Protection

1. **Encryption**: AES-256 for data at rest, TLS 1.3 for data in transit
2. **Access Controls**: Database-level row security
3. **Audit Logging**: Comprehensive audit trails for all operations
4. **Secure Downloads**: Time-limited, token-protected file downloads

### Monitoring

1. **Real-time Alerts**: Automated breach detection and notification
2. **Compliance Scoring**: Continuous compliance assessment
3. **Performance Monitoring**: API response time and system health
4. **Security Analytics**: Failed login attempts and unusual access patterns

## Data Subject Rights

### Right to Access (Article 15)

**Implementation:**
- Automated data collection from all systems
- Export in JSON, CSV, and FHIR R4 formats
- 30-day processing deadline with status tracking
- Secure download with 7-day expiration

**Process Flow:**
1. Patient submits access request
2. Step-up verification required
3. Background task collects data
4. Data packaged and encrypted
5. Download link sent to patient
6. Audit trail maintained

### Right to Erasure (Article 17)

**Implementation:**
- Automated deletion with legal exceptions
- Cascade deletion across all systems
- Anonymization for research data retention
- 30-day processing with admin approval

**Legal Exceptions:**
- Medical records retention (country-specific)
- Public health reporting requirements
- Legal claims and litigation holds
- Research with explicit consent

### Right to Data Portability (Article 20)

**Implementation:**
- FHIR R4 standard healthcare format
- Direct transfer API with OAuth 2.0
- Machine-readable structured data
- Metadata inclusion for processing history

## Consent Management

### Consent Types

| Type | Description | Legal Basis | Required |
|------|-------------|-------------|----------|
| **DATA_PROCESSING** | Essential healthcare services | Consent | ✅ |
| **RESEARCH** | Medical research participation | Consent | ❌ |
| **MARKETING** | Marketing communications | Consent | ❌ |
| **THIRD_PARTY_SHARING** | Data sharing with partners | Consent | ❌ |
| **ANALYTICS** | Usage analytics | Consent | ❌ |
| **AI_TRAINING** | ML model training | Consent | ❌ |
| **INTERNATIONAL_TRANSFER** | Data transfer outside EU | Consent | ❌ |
| **AUTOMATED_DECISION_MAKING** | AI-based decisions | Consent | ❌ |

### Consent Lifecycle

1. **Grant**: Explicit consent with detailed information
2. **Record**: Audit trail with IP, user agent, timestamp
3. **Verify**: Step-up authentication for sensitive consents
4. **Manage**: Update, revoke, or expire consents
5. **Proof**: Digital signature export for legal evidence

### Withdrawal Effects

- **Research**: Anonymize research data
- **Data Processing**: Initiate erasure request
- **AI Training**: Remove from training datasets
- **Marketing**: Unsubscribe from communications

## Data Retention

### Purpose-Based Retention

The implementation uses purpose-based retention rather than fixed periods:

```python
class RetentionService:
    def get_retention_period(self, country_code: str, data_category: str) -> str:
        # Returns country-specific retention period
        # Example: Germany medical records = "10 years after treatment completion"
    
    def calculate_retention_end_date(self, country_code: str, data_category: str, 
                                   start_date: datetime) -> Optional[datetime]:
        # Calculates exact retention end date
```

### Country-Specific Periods

| Country | Medical Records | Research Data | Epidemiological |
|---------|----------------|--------------|----------------|
| **DE** | 10 years | 15 years | 20 years |
| **FR** | 20 years | 15 years | 25 years |
| **UK** | 8 years | 20 years | 10 years |
| **IT** | 10 years | 25 years | 15 years |
| **ES** | 5 years | 15 years | 20 years |

### Automated Cleanup

- Daily tasks to identify expired data
- Soft delete with grace period
- Hard delete after legal requirements met
- Audit logging of all deletions

## Anonymization

### Techniques Implemented

1. **k-Anonymity**: k≥5 threshold for equivalence classes
2. **l-Diversity**: l≥3 diversity for sensitive attributes
3. **t-Closeness**: t≤0.1 distribution similarity
4. **Differential Privacy**: ε≤1.0 privacy budget
5. **Generalization**: Data generalization hierarchies
6. **Suppression**: Remove rare values
7. **Noise Addition**: Statistical noise for privacy

### Quality Assessment

```python
class AnonymizationResult:
    success: bool
    anonymized_records: int
    original_records: int
    k_anonymity_achieved: int
    l_diversity_achieved: int
    t_closeness_achieved: float
    re_identification_risk: str
    information_loss: float
```

### Proven Libraries

- **Presidio**: Microsoft's privacy protection library
- **pyanonymization**: Python anonymization toolkit
- **Custom algorithms**: EDPB-compliant implementations

## Breach Management

### Detection

Automated detection of:
- Unusual PHI access patterns (>100/hour)
- Mass data exports (>50/24h)
- Unauthorized access attempts (>20/15min)
- Failed login brute force (>10/5min)

### Notification Workflow

1. **Detection**: Automated breach identification
2. **Assessment**: Risk evaluation within 24 hours
3. **Authority Notification**: 72-hour deadline
4. **Subject Notification**: High-risk breaches only
5. **Documentation**: Complete breach register

### Breach Record

```python
class DataBreachRecord(Base):
    breach_id = Column(String(50), unique=True)
    breach_occurred = Column(DateTime(timezone=True))
    breach_discovered = Column(DateTime(timezone=True))
    nature = Column(String(100))
    categories_data = Column(JSON)
    number_affected = Column(Integer)
    high_risk = Column(Boolean)
    authority_notified = Column(DateTime(timezone=True))
    subjects_notified = Column(DateTime(timezone=True))
```

## Monitoring & Compliance

### Real-time Dashboard

- **Compliance Score**: Overall GDPR compliance percentage
- **Request Metrics**: Processing times and status distribution
- **Consent Analytics**: Grant/withdrawal rates by type
- **Security Status**: Active breaches and incidents
- **System Health**: API performance and storage usage

### Automated Alerts

- **Low Compliance Score**: <90% triggers warning
- **Overdue Requests**: Past deadline alerts
- **Security Breaches**: Immediate critical alerts
- **System Issues**: Performance and availability alerts

### Reporting

- **Daily Reports**: Compliance metrics and issues
- **Weekly Reports**: Trend analysis and summaries
- **Monthly Reports**: Comprehensive compliance status
- **Annual Reports**: Full GDPR compliance assessment

## Testing

### Unit Tests

```python
# Test coverage targets
- ConsentService: 95%
- DataSubjectRightsService: 95%
- AnonymizationService: 90%
- RetentionService: 90%
- API Endpoints: 90%
```

### Integration Tests

- **Complete Consent Workflow**: Grant → Check → Revoke → Verify
- **Data Access Request**: Request → Process → Download
- **Erasure Request**: Request → Approve → Delete → Verify
- **Breach Notification**: Detection → Assessment → Notification

### Compliance Tests

- **GDPR Article 15**: Access right compliance
- **GDPR Article 17**: Erasure right compliance
- **GDPR Article 20**: Portability right compliance
- **EDPB Guidelines**: Anonymization compliance
- **Country Laws**: Retention period compliance

### Performance Tests

- **Load Testing**: 1000 concurrent requests
- **Stress Testing**: Peak load scenarios
- **Endurance Testing**: 24-hour sustained load
- **Security Testing**: Penetration testing

## Deployment

### Environment Setup

```yaml
# Docker Compose Configuration
services:
  api:
    image: vitalstream-api:latest
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/vital
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
  
  worker:
    image: vitalstream-worker:latest
    command: celery -A app.celery_app worker --loglevel=info
  
  beat:
    image: vitalstream-beat:latest
    command: celery -A app.celery_app beat --loglevel=info
  
  redis:
    image: redis:7-alpine
  
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=vital
      - POSTGRES_USER=vital
      - POSTGRES_PASSWORD=secure_password
```

### Configuration

```python
# Production Settings
GDPR_SETTINGS = {
    "retention_service_enabled": True,
    "anonymization_enabled": True,
    "monitoring_enabled": True,
    "breach_detection_enabled": True,
    "step_up_authentication_required": True,
    "audit_logging_enabled": True,
    "encryption_at_rest": True,
    "encryption_in_transit": True
}
```

### Security Hardening

1. **Database**: Row-level security, encryption at rest
2. **API**: Rate limiting, input validation, CORS
3. **Storage**: Encrypted file storage, secure backups
4. **Network**: Firewall, VPN access, monitoring
5. **Authentication**: MFA, session management, password policies

## Maintenance

### Daily Tasks

- Expire old consents
- Clean up temporary files
- Process pending requests
- Check for security breaches
- Update compliance metrics

### Weekly Tasks

- Send weekly compliance reports
- Review and resolve alerts
- Update DPIA documentation
- Check system performance
- Backup critical data

### Monthly Tasks

- Comprehensive compliance review
- Update retention policies
- Review access logs
- Security assessment
- Performance optimization

### Annual Tasks

- Full GDPR compliance audit
- Legal review of policies
- DPO report preparation
- System security audit
- Documentation updates

### Monitoring Checklist

- [ ] Compliance score >90%
- [ ] No overdue requests
- [ ] No unresolved breaches
- [ ] System uptime >99%
- [ ] API response time <1s
- [ ] Error rate <1%
- [ ] Storage usage <80%
- [ ] Backup completion 100%

## Troubleshooting

### Common Issues

1. **Consent Not Recording**
   - Check database connection
   - Verify patient exists
   - Validate consent data format
   - Review audit logs

2. **Request Processing Delays**
   - Check Celery worker status
   - Verify Redis connection
   - Review queue length
   - Monitor system resources

3. **Breach Detection False Positives**
   - Review detection thresholds
   - Check access patterns
   - Update ML models
   - Adjust sensitivity settings

4. **Compliance Score Drop**
   - Review recent changes
   - Check data quality
   - Verify retention policies
   - Audit consent records

### Emergency Procedures

1. **Data Breach**
   - Immediate isolation
   - Assess impact
   - Notify authorities (72h)
   - Notify subjects (if high risk)
   - Document everything

2. **System Failure**
   - Activate backup systems
   - Notify stakeholders
   - Restore from backups
   - Investigate root cause
   - Implement prevention

3. **Compliance Violation**
   - Immediate investigation
   - Legal consultation
   - Corrective actions
   - Documentation
   - Prevention measures

## Support

### Contact Information

- **DPO**: dpo@vitalstream.com
- **Security Team**: security@vitalstream.com
- **Technical Support**: support@vitalstream.com
- **Legal Counsel**: legal@vitalstream.com

### External Resources

- [GDPR Official Text](https://gdpr-info.eu/)
- [EDPB Guidelines](https://edpb.europa.eu/)
- [ICO Guidance](https://ico.org.uk/)
- [FHIR R4 Standard](https://hl7.org/fhir/R4/)

### Training Materials

- GDPR Compliance Training
- Data Protection Training
- Security Awareness Training
- Incident Response Training

---

**Document Version**: 1.0  
**Approved By**: DPO Office
