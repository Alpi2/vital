# VitalStream API Documentation

## Base URL

```http
https://api.vitalstream.com/v1
```

## Authentication

All API requests require authentication using Bearer token:

```http
Authorization: Bearer YOUR_ACCESS_TOKEN
```

### Authentication Endpoints

#### Login (username/password)

Request examples (curl):

Demo user login:

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "doctor",
    "is_demo": true
  }'
```

Real user login:

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john.doe",
    "password": "SecurePass123!",
    "is_demo": false
  }'
```

Quick demo login by role:

```bash
curl -X POST http://localhost:8000/api/v1/auth/demo-login \
  -H "Content-Type: application/json" \
  -d '{"role": "doctor"}'
```

Response example:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900,
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "username": "doctor",
    "email": "doctor@vitalstream.demo",
    "first_name": "Sarah",
    "last_name": "Johnson",
    "role": "doctor",
    "is_demo": true
  }
}
```

---

## Endpoints

### Patients

#### Get All Patients

```http
GET /patients
```

**Response:**

```json
{
  "patients": [
    {
      "id": "P001",
      "name": "John Doe",
      "room": "ICU-1",
      "status": "active"
    }
  ]
}
```

#### Get Patient by ID

```http
GET /patients/{id}
```

**Response:**

```json
{
  "id": "P001",
  "name": "John Doe",
  "room": "ICU-1",
  "vital_signs": {
    "heart_rate": 75,
    "blood_pressure": "120/80",
    "spo2": 98
  }
}
```

### Alarms

#### Get Active Alarms

```http
GET /alarms?status=active
```

**Response:**

```json
{
  "alarms": [
    {
      "id": "A001",
      "patient_id": "P001",
      "severity": "critical",
      "message": "High heart rate",
      "timestamp": "2026-01-03T08:30:00Z"
    }
  ]
}
```

#### Acknowledge Alarm

```http
POST /alarms/{id}/acknowledge
```

**Request:**

```json
{
  "user_id": "U001",
  "notes": "Acknowledged by nurse"
}
```

**Response:**

```json
{
  "success": true,
  "alarm_id": "A001",
  "acknowledged_at": "2026-01-03T08:31:00Z"
}
```

---

## Error Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error

---

## Rate Limiting

- 1000 requests per hour per API key
- Rate limit headers included in response

## 🔗 Dependencies

- **Blocked By:** ticket:014b066c-6a96-465d-9aa1-711642e65572/9d706263-6e6e-4e98-819e-51a5cbec4fd1, ticket:014b066c-6a96-465d-9aa1-711642e65572/a5c25140-b6e5-47b3-a31f-6f3d3422840e, ticket:014b066c-6a96-465d-9aa1-711642e65572/2d582c13-88be-4c1d-9397-e2933ea93f12
- **Blocks:** Frontend login integration, protected routes
- **Related:** Audit logging, rate limiting

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-03
