from fastapi import APIRouter
from typing import List
from ..schemas.anomaly import AnomalyResponse, AnomalyCreate
from datetime import datetime
from ..limiter import limiter

"""
Anomalies API

Provides endpoints to list recent anomaly detections and to log new
anomaly events. `list_anomalies` is rate-limited to protect the API from
abuse.
"""

router = APIRouter()


@router.get(
    "/",
    response_model=List[AnomalyResponse],
    summary="List anomaly events",
    description="""Retrieve a list of detected anomaly events.
    
**Query Parameters (Future):**
- `patient_id`: Filter by patient ID
- `severity`: Filter by severity (critical, warning, info)
- `type`: Filter by anomaly type (arrhythmia, tachycardia, etc.)
- `resolved`: Filter by resolution status
- `start_date`: Filter by start date
- `end_date`: Filter by end date
- `limit`: Number of results to return
- `offset`: Pagination offset

**Returns:**
- Array of anomaly objects with full details
- Empty array if no anomalies found

**Rate Limit:**
- 100 requests per minute

**Example Response:**
```json
[
  {
    "id": 1,
    "patient_id": 1,
    "type": "arrhythmia",
    "severity": "critical",
    "confidence": 0.95,
    "timestamp": "2026-01-02T10:30:00Z",
    "resolved": false,
    "created_at": "2026-01-02T10:30:00Z"
  }
]
```
    """,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "patient_id": 1,
                            "type": "arrhythmia",
                            "severity": "critical",
                            "confidence": 0.95,
                            "timestamp": "2026-01-02T10:30:00Z",
                            "resolved": False,
                            "created_at": "2026-01-02T10:30:00Z"
                        }
                    ]
                }
            }
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {"detail": "Rate limit exceeded: 100 per 1 minute"}
                }
            }
        }
    }
)
@limiter.limit("100/minute")
async def list_anomalies():
    """List all anomaly events."""
    return []


@router.post(
    "/",
    response_model=AnomalyResponse,
    status_code=201,
    summary="Create/log an anomaly event",
    description="""Record a new anomaly detection event.
    
**Request Body:**
- `patient_id` (required): ID of the patient
- `type` (required): Type of anomaly (arrhythmia, tachycardia, bradycardia, etc.)
- `severity` (required): Severity level (critical, warning, info)
- `confidence` (optional): Detection confidence score (0.0-1.0)
- `description` (optional): Additional details about the anomaly

**Returns:**
- Created anomaly object with generated ID and timestamps
- Default `resolved` status is `false`

**Example Request:**
```json
{
  "patient_id": 1,
  "type": "arrhythmia",
  "severity": "critical",
  "confidence": 0.95,
  "description": "Irregular heart rhythm detected"
}
```

**Example Response:**
```json
{
  "id": 1,
  "patient_id": 1,
  "type": "arrhythmia",
  "severity": "critical",
  "confidence": 0.95,
  "description": "Irregular heart rhythm detected",
  "timestamp": "2026-01-02T10:30:00Z",
  "resolved": false,
  "created_at": "2026-01-02T10:30:00Z"
}
```
    """,
    responses={
        201: {
            "description": "Anomaly created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "patient_id": 1,
                        "type": "arrhythmia",
                        "severity": "critical",
                        "confidence": 0.95,
                        "timestamp": "2026-01-02T10:30:00Z",
                        "resolved": False,
                        "created_at": "2026-01-02T10:30:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input data",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid anomaly data"}
                }
            }
        },
        404: {
            "description": "Patient not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Patient with ID 1 not found"}
                }
            }
        }
    }
)
async def create_anomaly(payload: AnomalyCreate):
    """Create/log a new anomaly event."""
    now = datetime.utcnow()
    return {
        "id": 1,
        **payload.dict(),
        "timestamp": now,
        "created_at": now,
        "resolved": False,
    }
