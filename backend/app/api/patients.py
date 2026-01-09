from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..schemas.patient import PatientRead, PatientCreate
from ..schemas.patient import PatientBase

"""
Patients API

Endpoints:
 - GET / : list patients
 - POST / : create a new patient

The endpoints are intentionally simple stubs for now; they return
structures compatible with the Pydantic schemas and include descriptive
docstrings for OpenAPI generation.
"""

router = APIRouter()


@router.get(
    "/",
    response_model=List[PatientRead],
    summary="List all patients",
    description="""Retrieve a list of all patient records.
    
**Query Parameters:**
- Future: Add filtering by name, age, status
- Future: Add pagination support
- Future: Add sorting options

**Returns:**
- Array of patient objects with full details
- Empty array if no patients exist

**Example Response:**
```json
[
  {
    "id": 1,
    "name": "John Doe",
    "age": 45,
    "gender": "male",
    "medical_record_number": "MRN-12345",
    "created_at": "2026-01-02T10:00:00Z"
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
                            "name": "John Doe",
                            "age": 45,
                            "gender": "male",
                            "medical_record_number": "MRN-12345",
                            "created_at": "2026-01-02T10:00:00Z"
                        }
                    ]
                }
            }
        }
    }
)
async def list_patients():
    """List all patients."""
    return []


@router.post(
    "/",
    response_model=PatientRead,
    status_code=201,
    summary="Create a new patient",
    description="""Create a new patient record in the system.
    
**Request Body:**
- `name` (required): Patient's full name
- `age` (required): Patient's age in years
- `gender` (optional): Patient's gender
- `medical_record_number` (optional): Unique medical record identifier

**Returns:**
- Created patient object with generated ID and timestamp

**Example Request:**
```json
{
  "name": "John Doe",
  "age": 45,
  "gender": "male",
  "medical_record_number": "MRN-12345"
}
```

**Example Response:**
```json
{
  "id": 1,
  "name": "John Doe",
  "age": 45,
  "gender": "male",
  "medical_record_number": "MRN-12345",
  "created_at": "2026-01-02T10:00:00Z"
}
```
    """,
    responses={
        201: {
            "description": "Patient created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "name": "John Doe",
                        "age": 45,
                        "gender": "male",
                        "medical_record_number": "MRN-12345",
                        "created_at": "2026-01-02T10:00:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input data",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid patient data"}
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "age"],
                                "msg": "ensure this value is greater than 0",
                                "type": "value_error.number.not_gt"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def create_patient(payload: PatientCreate):
    """Create a new patient record."""
    return {"id": 1, **payload.dict(), "created_at": None}
