from fastapi import APIRouter, Response

router = APIRouter()


@router.get("/generate/{patient_id}/{session_id}")
async def generate_report(patient_id: int, session_id: int):
    # placeholder: generate PDF/report for session
    pdf_bytes = b"%PDF-1.4\n%EOF"
    return Response(content=pdf_bytes, media_type="application/pdf")
