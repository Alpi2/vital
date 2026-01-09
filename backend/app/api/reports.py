from fastapi import APIRouter, Response

"""
Reports API

Endpoints to generate patient/session PDF reports. Currently returns
a placeholder PDF byte stream for development and testing.
"""

router = APIRouter()


@router.get(
    "/generate/{patient_id}/{session_id}",
    summary="Generate patient session report",
    description="""Generate a comprehensive PDF report for a patient session.
    
**Path Parameters:**
- `patient_id` (required): ID of the patient
- `session_id` (required): ID of the monitoring session

**Report Contents:**
- Patient demographic information
- Session details (start time, duration, status)
- ECG waveform visualization
- Detected anomalies with timestamps
- Statistical analysis (heart rate, variability, etc.)
- Summary and recommendations

**Returns:**
- PDF file as binary stream
- Content-Type: application/pdf
- Filename in Content-Disposition header

**Example Usage:**
```bash
curl -O -J http://localhost:8000/api/v1/reports/generate/1/123
```

**Note:** Current implementation returns a minimal stub PDF.
    """,
    responses={
        200: {
            "description": "PDF report generated successfully",
            "content": {
                "application/pdf": {
                    "schema": {
                        "type": "string",
                        "format": "binary"
                    }
                }
            },
            "headers": {
                "Content-Disposition": {
                    "description": "Attachment filename",
                    "schema": {
                        "type": "string",
                        "example": "attachment; filename=patient_1_session_123_report.pdf"
                    }
                }
            }
        },
        404: {
            "description": "Patient or session not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Patient with ID 1 or session 123 not found"}
                }
            }
        },
        500: {
            "description": "Report generation failed",
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to generate PDF report"}
                }
            }
        }
    },
    response_class=Response
)
async def generate_report(patient_id: int, session_id: int):
    """Generate a PDF report for a patient session."""
    pdf_bytes = b"%PDF-1.4\n%EOF"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=patient_{patient_id}_session_{session_id}_report.pdf"
        }
    )
