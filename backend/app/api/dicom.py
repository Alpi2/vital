"""
DICOM API endpoints for VitalStream

This module provides REST API endpoints for DICOM operations including:
- File upload and storage
- File retrieval and download
- ECG waveform extraction
- DICOM metadata querying
- File management (list, delete)
- Health checks

All operations are performed asynchronously using the gRPC DICOM service.
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Response, Query
from fastapi.concurrency import run_in_threadpool
from typing import Optional, List
import logging
import asyncio
from datetime import datetime

from app.services.grpc.dicom_client import DICOMGRPCClient, dicom_client
from app.security.dependencies import get_current_user
from app.models.user import User
from app.schemas.dicom import (
    DICOMUploadResponse,
    DICOMMetadataResponse,
    WaveformResponse,
    DICOMQueryResponse,
    DICOMListResponse,
    DICOMDeleteResponse,
    HealthCheckResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/dicom", tags=["DICOM"])

@router.post("/upload", response_model=DICOMUploadResponse)
async def upload_dicom(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Upload DICOM file to storage
    
    Args:
        file: DICOM file to upload
        patient_id: Optional patient identifier
        study_instance_uid: Optional study instance UID
        series_instance_uid: Optional series instance UID
        current_user: Authenticated user
        
    Returns:
        DICOMUploadResponse: Upload result with metadata
        
    Raises:
        HTTPException: For validation or processing errors
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        if not file.filename.lower().endswith(('.dcm', '.dicom')):
            raise HTTPException(
                status_code=400, 
                detail="Only DICOM files (.dcm, .dicom) are allowed"
            )
        
        # Read file with memory management for large files
        logger.info(f"📁 Reading DICOM file: {file.filename} ({file.content_type})")
        
        # Use run_in_threadpool to avoid blocking the event loop
        file_data = await run_in_threadpool(file.read)
        
        if len(file_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        if len(file_data) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(
                status_code=400, 
                detail="File too large (max 100MB)"
            )
        
        # Store via gRPC service
        metadata = await dicom_client.store_dicom(
            file_data=file_data,
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid
        )
        
        logger.info(
            f"🏥 DICOM uploaded successfully: {metadata['dicom_id']} "
            f"by user {current_user.id} ({metadata['file_size']/1024:.1f}KB)"
        )
        
        return DICOMUploadResponse(
            status="success",
            dicom_id=metadata['dicom_id'],
            patient_id=metadata['patient_id'],
            patient_name=metadata['patient_name'],
            study_date=metadata['study_date'],
            modality=metadata['modality'],
            file_size=metadata['file_size'],
            created_at=metadata['created_at'],
            study_instance_uid=metadata['study_instance_uid'],
            series_instance_uid=metadata['series_instance_uid'],
            sop_instance_uid=metadata['sop_instance_uid'],
            has_waveform='waveform_data' in metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DICOM upload error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to upload DICOM file"
        )

@router.get("/{dicom_id}")
async def get_dicom(
    dicom_id: str,
    include_metadata: bool = Query(default=False, description="Include full metadata"),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve DICOM file
    
    Args:
        dicom_id: DICOM file identifier
        include_metadata: Whether to include full metadata
        current_user: Authenticated user
        
    Returns:
        Response: DICOM file data with appropriate headers
        
    Raises:
        HTTPException: If file not found or access denied
    """
    try:
        if not dicom_id:
            raise HTTPException(status_code=400, detail="DICOM ID is required")
        
        logger.info(f"📥 Retrieving DICOM file: {dicom_id}")
        
        result = await dicom_client.get_dicom(
            dicom_id=dicom_id,
            include_metadata=include_metadata
        )
        
        # Prepare response headers
        headers = {
            "Content-Type": "application/dicom",
            "Content-Disposition": f"attachment; filename=\"{dicom_id}.dcm\"",
            "Content-Length": str(result['file_size']),
        }
        
        # Add metadata headers if requested
        if include_metadata and 'metadata' in result:
            metadata = result['metadata']
            headers.update({
                "X-DICOM-Patient-ID": metadata.get('patient_id', ''),
                "X-DICOM-Patient-Name": metadata.get('patient_name', ''),
                "X-DICOM-Modality": metadata.get('modality', ''),
                "X-DICOM-Study-Date": metadata.get('study_date', ''),
            })
        
        logger.info(f"✅ DICOM retrieved: {dicom_id} ({result['file_size']/1024:.1f}KB)")
        
        return Response(
            content=result['file_data'],
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DICOM retrieval error: {str(e)}")
        raise HTTPException(
            status_code=404, 
            detail="DICOM file not found"
        )

@router.get("/{dicom_id}/metadata", response_model=DICOMMetadataResponse)
async def get_dicom_metadata(
    dicom_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get DICOM file metadata without downloading the file
    
    Args:
        dicom_id: DICOM file identifier
        current_user: Authenticated user
        
    Returns:
        DICOMMetadataResponse: DICOM metadata
        
    Raises:
        HTTPException: If file not found
    """
    try:
        if not dicom_id:
            raise HTTPException(status_code=400, detail="DICOM ID is required")
        
        logger.info(f"📋 Getting DICOM metadata: {dicom_id}")
        
        result = await dicom_client.get_dicom(dicom_id=dicom_id, include_metadata=True)
        
        if 'metadata' not in result:
            raise HTTPException(status_code=404, detail="Metadata not found")
        
        metadata = result['metadata']
        
        logger.info(f"✅ DICOM metadata retrieved: {dicom_id}")
        
        return DICOMMetadataResponse(
            dicom_id=dicom_id,
            patient_id=metadata['patient_id'],
            patient_name=metadata['patient_name'],
            study_date=metadata['study_date'],
            modality=metadata['modality'],
            file_size=result['file_size'],
            created_at=metadata['created_at'],
            study_instance_uid=metadata['study_instance_uid'],
            series_instance_uid=metadata['series_instance_uid'],
            sop_instance_uid=metadata['sop_instance_uid']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DICOM metadata error: {str(e)}")
        raise HTTPException(
            status_code=404, 
            detail="DICOM metadata not found"
        )

@router.get("/{dicom_id}/waveform", response_model=WaveformResponse)
async def get_dicom_waveform(
    dicom_id: str,
    channels: Optional[List[str]] = Query(default=None, description="Specific channels to extract"),
    current_user: User = Depends(get_current_user)
):
    """
    Extract ECG waveform from DICOM file
    
    Args:
        dicom_id: DICOM file identifier
        channels: Optional list of specific channels to extract
        current_user: Authenticated user
        
    Returns:
        WaveformResponse: Extracted waveform data
        
    Raises:
        HTTPException: If file not found or no waveform data
    """
    try:
        if not dicom_id:
            raise HTTPException(status_code=400, detail="DICOM ID is required")
        
        logger.info(f"📈 Extracting waveform from DICOM: {dicom_id}")
        
        waveform = await dicom_client.extract_waveform(
            dicom_id=dicom_id,
            channel_names=channels
        )
        
        if not waveform:
            raise HTTPException(
                status_code=404, 
                detail="No waveform data found in DICOM file"
            )
        
        logger.info(
            f"✅ Waveform extracted: {dicom_id} "
            f"({waveform['num_channels']} channels, {waveform['num_samples']} samples)"
        )
        
        return WaveformResponse(
            dicom_id=dicom_id,
            num_channels=waveform['num_channels'],
            num_samples=waveform['num_samples'],
            sampling_frequency=waveform['sampling_frequency'],
            samples=waveform['samples'],
            channel_names=waveform['channel_names'],
            sensitivity=waveform['sensitivity'],
            baseline=waveform['baseline'],
            units=waveform['units'],
            bits_allocated=waveform['bits_allocated'],
            bits_stored=waveform['bits_stored'],
            is_signed=waveform['is_signed']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Waveform extraction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to extract waveform"
        )

@router.get("/query", response_model=DICOMQueryResponse)
async def query_dicom(
    patient_id: Optional[str] = Query(default=None, description="Filter by patient ID"),
    study_instance_uid: Optional[str] = Query(default=None, description="Filter by study instance UID"),
    modality: Optional[str] = Query(default=None, description="Filter by modality"),
    date_from: Optional[str] = Query(default=None, description="Filter by date from (ISO format)"),
    date_to: Optional[str] = Query(default=None, description="Filter by date to (ISO format)"),
    page_size: int = Query(default=50, ge=1, le=1000, description="Number of results per page"),
    page_token: Optional[str] = Query(default=None, description="Pagination token"),
    current_user: User = Depends(get_current_user)
):
    """
    Query DICOM files with filtering and pagination
    
    Args:
        patient_id: Filter by patient ID
        study_instance_uid: Filter by study instance UID
        modality: Filter by modality (e.g., "ECG", "CT", "MR")
        date_from: Filter by date from (ISO format)
        date_to: Filter by date to (ISO format)
        page_size: Number of results per page
        page_token: Pagination token
        current_user: Authenticated user
        
    Returns:
        DICOMQueryResponse: Query results with pagination
        
    Raises:
        HTTPException: For query errors
    """
    try:
        logger.info(f"🔍 Querying DICOM files: patient_id={patient_id}, modality={modality}")
        
        result = await dicom_client.query_dicom(
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            modality=modality,
            date_from=date_from,
            date_to=date_to,
            page_size=page_size,
            page_token=page_token
        )
        
        logger.info(f"✅ DICOM query completed: {len(result['dicom_files'])} results")
        
        return DICOMQueryResponse(
            dicom_files=result['dicom_files'],
            total_count=result['total_count'],
            page_size=page_size,
            next_page_token=result['next_page_token']
        )
        
    except Exception as e:
        logger.error(f"❌ DICOM query error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to query DICOM files"
        )

@router.get("/list", response_model=DICOMListResponse)
async def list_dicom(
    patient_id: Optional[str] = Query(default=None, description="Filter by patient ID"),
    modality: Optional[str] = Query(default=None, description="Filter by modality"),
    page_size: int = Query(default=50, ge=1, le=1000, description="Number of results per page"),
    page_token: Optional[str] = Query(default=None, description="Pagination token"),
    current_user: User = Depends(get_current_user)
):
    """
    List DICOM files with filtering and pagination
    
    Args:
        patient_id: Filter by patient ID
        modality: Filter by modality
        page_size: Number of results per page
        page_token: Pagination token
        current_user: Authenticated user
        
    Returns:
        DICOMListResponse: List results with pagination
        
    Raises:
        HTTPException: For list errors
    """
    try:
        logger.info(f"📋 Listing DICOM files: patient_id={patient_id}, modality={modality}")
        
        result = await dicom_client.list_dicom(
            patient_id=patient_id,
            modality=modality,
            page_size=page_size,
            page_token=page_token
        )
        
        logger.info(f"✅ DICOM list completed: {len(result['dicom_files'])} results")
        
        return DICOMListResponse(
            dicom_files=result['dicom_files'],
            total_count=result['total_count'],
            page_size=page_size,
            next_page_token=result['next_page_token']
        )
        
    except Exception as e:
        logger.error(f"❌ DICOM list error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to list DICOM files"
        )

@router.delete("/{dicom_id}", response_model=DICOMDeleteResponse)
async def delete_dicom(
    dicom_id: str,
    permanent: bool = Query(default=False, description="Permanently delete (vs move to trash)"),
    current_user: User = Depends(get_current_user)
):
    """
    Delete DICOM file
    
    Args:
        dicom_id: DICOM file identifier
        permanent: If False, move to trash; if True, permanently delete
        current_user: Authenticated user
        
    Returns:
        DICOMDeleteResponse: Deletion result
        
    Raises:
        HTTPException: If file not found or deletion fails
    """
    try:
        if not dicom_id:
            raise HTTPException(status_code=400, detail="DICOM ID is required")
        
        logger.info(f"🗑️ Deleting DICOM file: {dicom_id} (permanent={permanent})")
        
        result = await dicom_client.delete_dicom(
            dicom_id=dicom_id,
            permanent=permanent
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=result['message']
            )
        
        logger.info(f"✅ DICOM deleted: {dicom_id}")
        
        return DICOMDeleteResponse(
            dicom_id=dicom_id,
            success=result['success'],
            message=result['message'],
            permanent=permanent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DICOM deletion error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to delete DICOM file"
        )

@router.get("/health", response_model=HealthCheckResponse)
async def dicom_health_check():
    """
    Check DICOM service health
    
    Returns:
        HealthCheckResponse: Health status information
        
    Raises:
        HTTPException: If health check fails
    """
    try:
        logger.info("🏥 Performing DICOM service health check")
        
        result = await dicom_client.health_check()
        
        return HealthCheckResponse(
            status=result['status'],
            service=result['service'],
            latency_ms=result.get('latency_ms', 0),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"❌ DICOM health check error: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            service="dicom-grpc",
            latency_ms=0,
            error=str(e)
        )

@router.get("/{dicom_id}/preview")
async def get_dicom_preview(
    dicom_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get DICOM file preview (metadata only, no file data)
    
    Args:
        dicom_id: DICOM file identifier
        current_user: Authenticated user
        
    Returns:
        Dict: Preview information
        
    Raises:
        HTTPException: If file not found
    """
    try:
        if not dicom_id:
            raise HTTPException(status_code=400, detail="DICOM ID is required")
        
        logger.info(f"👁️ Getting DICOM preview: {dicom_id}")
        
        # Get metadata without file data
        result = await dicom_client.get_dicom(dicom_id=dicom_id, include_metadata=True)
        
        if 'metadata' not in result:
            raise HTTPException(status_code=404, detail="DICOM file not found")
        
        metadata = result['metadata']
        
        # Check if waveform is available
        has_waveform = await dicom_client.extract_waveform(dicom_id)
        
        preview = {
            'dicom_id': dicom_id,
            'patient_id': metadata['patient_id'],
            'patient_name': metadata['patient_name'],
            'study_date': metadata['study_date'],
            'modality': metadata['modality'],
            'file_size': result['file_size'],
            'created_at': metadata['created_at'],
            'study_instance_uid': metadata['study_instance_uid'],
            'series_instance_uid': metadata['series_instance_uid'],
            'sop_instance_uid': metadata['sop_instance_uid'],
            'has_waveform': has_waveform is not None,
        }
        
        logger.info(f"✅ DICOM preview retrieved: {dicom_id}")
        
        return preview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DICOM preview error: {str(e)}")
        raise HTTPException(
            status_code=404, 
            detail="DICOM file not found"
        )
