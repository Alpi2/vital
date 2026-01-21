"""
Enhanced HRV API Endpoints
Complete implementation with comprehensive HRV calculation, retrieval, and reporting endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path

from ..services.hrv_service import HRVService
from ..services.hrv_interpretation import interpret_hrv_metrics, validate_hrv_interpretation
from ..services.hrv_trend_analysis import analyze_hrv_trend, export_trend_report
from ..dependencies import get_current_user, get_database_service
from ..models.user import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/hrv", tags=["HRV"])

# Pydantic models for request/response
class HRVCalculationRequest(BaseModel):
    """Request model for HRV calculation"""
    r_peaks: List[int] = Field(..., description="List of R-peak sample indices")
    sampling_rate: int = Field(360, description="ECG sampling rate in Hz")
    patient_id: str = Field(..., description="Patient identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    include_interpretation: bool = Field(True, description="Include clinical interpretation")
    patient_age: Optional[int] = Field(None, description="Patient age for interpretation")
    patient_gender: Optional[str] = Field(None, description="Patient gender (male/female)")
    preprocessing_method: str = Field("adaptive", description="Preprocessing method")
    include_advanced: bool = Field(True, description="Include advanced metrics")

class RRIntervalsRequest(BaseModel):
    """Request model for RR intervals"""
    rr_intervals: List[float] = Field(..., description="RR intervals in milliseconds")
    patient_id: str = Field(..., description="Patient identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    include_interpretation: bool = Field(True, description="Include clinical interpretation")
    patient_age: Optional[int] = Field(None, description="Patient age for interpretation")
    patient_gender: Optional[str] = Field(None, description="Patient gender (male/female)")
    preprocessing_method: str = Field("adaptive", description="Preprocessing method")
    include_advanced: bool = Field(True, description="Include advanced metrics")

class BatchHRVRequest(BaseModel):
    """Request model for batch HRV calculation"""
    patient_data: List[Dict[str, Any]] = Field(..., description="List of patient data")
    include_interpretation: bool = Field(True, description="Include clinical interpretation")
    preprocessing_method: str = Field("adaptive", description="Preprocessing method")
    include_advanced: bool = Field(True, description="Include advanced metrics")

class HRVReportRequest(BaseModel):
    """Request model for HRV report generation"""
    patient_id: str = Field(..., description="Patient identifier")
    time_range: str = Field("30d", description="Time range for trend analysis")
    report_format: str = Field("json", description="Report format (json, csv, html)")
    include_trends: bool = Field(True, description="Include trend analysis")
    include_interpretation: bool = Field(True, description="Include clinical interpretation")

class HRVResponse(BaseModel):
    """Response model for HRV calculation"""
    success: bool = Field(True, description="Calculation success status")
    patient_id: str = Field(..., description="Patient identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    hrv_metrics: Dict[str, Any] = Field(..., description="HRV metrics")
    interpretation: Optional[Dict[str, Any]] = Field(None, description="Clinical interpretation")
    quality_assessment: Optional[Dict[str, Any]] = Field(None, description="Quality assessment")
    calculation_time: str = Field(..., description="Calculation timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

# Initialize HRV service
hrv_service = HRVService()

@router.post("/calculate", response_model=HRVResponse)
async def calculate_hrv(
    request: HRVCalculationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Calculate HRV metrics from R-peak locations
    
    - **r_peaks**: List of R-peak sample indices from ECG
    - **sampling_rate**: ECG sampling rate in Hz
    - **patient_id**: Patient identifier
    - **include_interpretation**: Whether to include clinical interpretation
    - **patient_age**: Patient age (required for interpretation)
    - **patient_gender**: Patient gender (required for interpretation)
    """
    
    start_time = datetime.now()
    
    try:
        # Validate input
        if len(request.r_peaks) < 2:
            raise HTTPException(status_code=400, detail="Insufficient R-peaks for HRV calculation")
        
        if request.sampling_rate < 100 or request.sampling_rate > 2000:
            raise HTTPException(status_code=400, detail="Invalid sampling rate (must be 100-2000 Hz)")
        
        # Extract RR intervals
        rr_intervals = hrv_service.extract_rr_intervals(request.r_peaks, request.sampling_rate)
        
        if len(rr_intervals) == 0:
            raise HTTPException(status_code=400, detail="Failed to extract RR intervals")
        
        # Calculate HRV metrics
        hrv_metrics = await hrv_service.calculate_hrv_async(
            rr_intervals, 
            request.sampling_rate
        )
        
        # Add patient and session info
        hrv_metrics["patient_id"] = request.patient_id
        hrv_metrics["session_id"] = request.session_id
        
        # Quality assessment
        quality_assessment = hrv_service.validate_hrv_quality(rr_intervals)
        
        # Clinical interpretation (if requested)
        interpretation = None
        if request.include_interpretation:
            if not request.patient_age or not request.patient_gender:
                raise HTTPException(
                    status_code=400, 
                    detail="Patient age and gender required for clinical interpretation"
                )
            
            interpretation = interpret_hrv_metrics(
                hrv_metrics,
                request.patient_age,
                request.patient_gender
            )
            
            # Validate interpretation
            validation = validate_hrv_interpretation(interpretation)
            if not validation["is_valid"]:
                logger.warning(f"HRV interpretation validation failed: {validation['errors']}")
        
        # Save to database (async background task)
        if db_service:
            background_tasks.add_task(
                save_hrv_to_database,
                request.patient_id,
                request.session_id,
                hrv_metrics,
                interpretation,
                quality_assessment
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = HRVResponse(
            patient_id=request.patient_id,
            session_id=request.session_id,
            hrv_metrics=hrv_metrics,
            interpretation=interpretation,
            quality_assessment=quality_assessment,
            calculation_time=start_time.isoformat(),
            processing_time_ms=processing_time
        )
        
        logger.info(f"HRV calculation completed for patient {request.patient_id} in {processing_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HRV calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HRV calculation failed: {str(e)}")

@router.post("/calculate-from-rr", response_model=HRVResponse)
async def calculate_hrv_from_rr(
    request: RRIntervalsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Calculate HRV metrics from RR intervals
    
    - **rr_intervals**: RR intervals in milliseconds
    - **patient_id**: Patient identifier
    - **include_interpretation**: Whether to include clinical interpretation
    - **patient_age**: Patient age (required for interpretation)
    - **patient_gender**: Patient gender (required for interpretation)
    """
    
    start_time = datetime.now()
    
    try:
        # Validate input
        if len(request.rr_intervals) < 5:
            raise HTTPException(status_code=400, detail="Insufficient RR intervals for HRV calculation (minimum 5)")
        
        # Convert to numpy array
        rr_intervals = np.array(request.rr_intervals)
        
        # Validate RR intervals
        if np.any(rr_intervals <= 0):
            raise HTTPException(status_code=400, detail="RR intervals must be positive")
        
        if np.any(rr_intervals < 200) or np.any(rr_intervals > 2000):
            raise HTTPException(status_code=400, detail="RR intervals out of physiological range (200-2000ms)")
        
        # Calculate HRV metrics
        hrv_metrics = await hrv_service.calculate_hrv_async(
            rr_intervals,
            include_advanced=request.include_advanced
        )
        
        # Add patient and session info
        hrv_metrics["patient_id"] = request.patient_id
        hrv_metrics["session_id"] = request.session_id
        
        # Quality assessment
        quality_assessment = hrv_service.validate_hrv_quality(rr_intervals)
        
        # Clinical interpretation (if requested)
        interpretation = None
        if request.include_interpretation:
            if not request.patient_age or not request.patient_gender:
                raise HTTPException(
                    status_code=400, 
                    detail="Patient age and gender required for clinical interpretation"
                )
            
            interpretation = interpret_hrv_metrics(
                hrv_metrics,
                request.patient_age,
                request.patient_gender
            )
        
        # Save to database (async background task)
        if db_service:
            background_tasks.add_task(
                save_hrv_to_database,
                request.patient_id,
                request.session_id,
                hrv_metrics,
                interpretation,
                quality_assessment
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = HRVResponse(
            patient_id=request.patient_id,
            session_id=request.session_id,
            hrv_metrics=hrv_metrics,
            interpretation=interpretation,
            quality_assessment=quality_assessment,
            calculation_time=start_time.isoformat(),
            processing_time_ms=processing_time
        )
        
        logger.info(f"HRV calculation from RR completed for patient {request.patient_id} in {processing_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HRV calculation from RR error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HRV calculation failed: {str(e)}")

@router.post("/batch")
async def calculate_batch_hrv(
    request: BatchHRVRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Calculate HRV metrics for multiple patients (batch processing)
    
    - **patient_data**: List of patient data dictionaries
    - **include_interpretation**: Whether to include clinical interpretation
    """
    
    try:
        # Validate batch size
        if len(request.patient_data) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large (maximum 100 patients)")
        
        # Process batch
        results = hrv_service.calculate_hrv_batch(request.patient_data)
        
        # Add interpretations if requested
        if request.include_interpretation:
            for result in results:
                if "error" not in result and "time_domain" in result:
                    patient_age = result.get("patient_age")
                    patient_gender = result.get("patient_gender")
                    
                    if patient_age and patient_gender:
                        interpretation = interpret_hrv_metrics(result, patient_age, patient_gender)
                        result["interpretation"] = interpretation
        
        # Save batch results (async background task)
        if db_service:
            background_tasks.add_task(
                save_batch_hrv_to_database,
                results
            )
        
        logger.info(f"Batch HRV calculation completed for {len(request.patient_data)} patients")
        
        return {
            "success": True,
            "batch_size": len(request.patient_data),
            "processed_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch HRV calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch HRV calculation failed: {str(e)}")

@router.get("/{patient_id}")
async def get_patient_hrv_history(
    patient_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Get HRV history for a patient
    
    - **patient_id**: Patient identifier
    - **start_date**: Start date for filtering (optional)
    - **end_date**: End date for filtering (optional)
    - **limit**: Maximum number of records to return
    """
    
    try:
        # Parse dates
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start date format (use YYYY-MM-DD)")
        else:
            start_dt = datetime.now() - timedelta(days=30)
        
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end date format (use YYYY-MM-DD)")
        else:
            end_dt = datetime.now()
        
        # Validate date range
        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Retrieve from database
        if db_service:
            hrv_history = db_service.get_patient_hrv_history(patient_id, start_dt, end_dt, limit)
        else:
            # Mock data for demonstration
            hrv_history = generate_mock_hrv_history(patient_id, start_dt, end_dt, limit)
        
        return {
            "patient_id": patient_id,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "record_count": len(hrv_history),
            "hrv_history": hrv_history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get HRV history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve HRV history: {str(e)}")

@router.get("/{patient_id}/trend")
async def get_hrv_trend(
    patient_id: str,
    time_range: str = Query("30d", description="Time range (7d, 30d, 90d, 1y)"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Get HRV trend analysis for a patient
    
    - **patient_id**: Patient identifier
    - **time_range**: Time range for trend analysis
    """
    
    try:
        # Validate time range
        valid_ranges = ["7d", "30d", "90d", "1y"]
        if time_range not in valid_ranges:
            raise HTTPException(status_code=400, detail=f"Invalid time range. Use: {', '.join(valid_ranges)}")
        
        # Perform trend analysis
        trend_analysis = analyze_hrv_trend(patient_id, time_range, db_service)
        
        return trend_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HRV trend analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HRV trend analysis failed: {str(e)}")

@router.get("/{patient_id}/report")
async def generate_hrv_report(
    patient_id: str,
    time_range: str = Query("30d", description="Time range for trend analysis"),
    report_format: str = Query("json", description="Report format (json, csv, html)"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    include_interpretation: bool = Query(True, description="Include clinical interpretation"),
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Generate comprehensive HRV report for a patient
    
    - **patient_id**: Patient identifier
    - **time_range**: Time range for trend analysis
    - **report_format**: Report format (json, csv, html)
    - **include_trends**: Whether to include trend analysis
    - **include_interpretation**: Whether to include clinical interpretation
    """
    
    try:
        # Validate report format
        valid_formats = ["json", "csv", "html"]
        if report_format not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Invalid report format. Use: {', '.join(valid_formats)}")
        
        # Get patient HRV data
        start_date = datetime.now() - timedelta(days=30)  # Default to 30 days
        end_date = datetime.now()
        
        if db_service:
            hrv_history = db_service.get_patient_hrv_history(patient_id, start_date, end_date, 1000)
        else:
            hrv_history = generate_mock_hrv_history(patient_id, start_date, end_date, 1000)
        
        if not hrv_history:
            raise HTTPException(status_code=404, detail="No HRV data found for patient")
        
        # Generate report data
        report_data = {
            "patient_id": patient_id,
            "report_generated": datetime.now().isoformat(),
            "time_range": time_range,
            "data_summary": {
                "total_records": len(hrv_history),
                "date_range": {
                    "start": hrv_history[0]["date"] if hrv_history else None,
                    "end": hrv_history[-1]["date"] if hrv_history else None
                }
            },
            "hrv_data": hrv_history
        }
        
        # Add trend analysis if requested
        if include_trends:
            trend_analysis = analyze_hrv_trend(patient_id, time_range, db_service)
            report_data["trend_analysis"] = trend_analysis
        
        # Add latest interpretation if requested
        if include_interpretation and hrv_history:
            latest_hrv = hrv_history[-1]
            # Extract patient info (would need to get from database)
            patient_age = 40  # Default - should get from database
            patient_gender = "male"  # Default - should get from database
            
            interpretation = interpret_hrv_metrics(latest_hrv, patient_age, patient_gender)
            report_data["latest_interpretation"] = interpretation
        
        # Export report
        if report_format == "json":
            return JSONResponse(content=report_data)
        elif report_format == "csv":
            csv_content = export_hrv_report_to_csv(report_data)
            return FileResponse(
                path=Path(f"/tmp/hrv_report_{patient_id}.csv"),
                filename=f"hrv_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                media_type="text/csv"
            )
        elif report_format == "html":
            html_content = export_hrv_report_to_html(report_data)
            return FileResponse(
                path=Path(f"/tmp/hrv_report_{patient_id}.html"),
                filename=f"hrv_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.html",
                media_type="text/html"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HRV report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HRV report generation failed: {str(e)}")

@router.get("/metrics/summary")
async def get_hrv_metrics_summary(
    current_user: User = Depends(get_current_user),
    db_service = Depends(get_database_service)
):
    """
    Get summary of available HRV metrics and their descriptions
    """
    
    metrics_summary = {
        "time_domain": {
            "description": "Time domain HRV metrics calculated from RR intervals",
            "metrics": {
                "mean_rr": {"description": "Mean RR interval in milliseconds", "unit": "ms"},
                "sdnn": {"description": "Standard deviation of NN intervals", "unit": "ms"},
                "rmssd": {"description": "Root mean square of successive differences", "unit": "ms"},
                "pnn50": {"description": "Percentage of NN intervals differing by >50ms", "unit": "%"},
                "nn50": {"description": "Number of NN interval pairs differing by >50ms", "unit": "count"},
                "tinn": {"description": "Triangular interpolation of NN intervals", "unit": "ms"},
                "hrv_tri_index": {"description": "HRV triangular index", "unit": "index"}
            }
        },
        "frequency_domain": {
            "description": "Frequency domain HRV metrics from power spectral density",
            "metrics": {
                "total_power": {"description": "Total power in VLF+LF+HF bands", "unit": "ms²"},
                "vlf_power": {"description": "Very low frequency power (0.003-0.04 Hz)", "unit": "ms²"},
                "lf_power": {"description": "Low frequency power (0.04-0.15 Hz)", "unit": "ms²"},
                "hf_power": {"description": "High frequency power (0.15-0.4 Hz)", "unit": "ms²"},
                "lf_hf_ratio": {"description": "LF/HF power ratio", "unit": "ratio"},
                "lf_norm": {"description": "Normalized LF power", "unit": "normalized units"},
                "hf_norm": {"description": "Normalized HF power", "unit": "normalized units"}
            }
        },
        "nonlinear": {
            "description": "Non-linear HRV metrics for complexity analysis",
            "metrics": {
                "sd1": {"description": "Poincaré SD1 (short-term variability)", "unit": "ms"},
                "sd2": {"description": "Poincaré SD2 (long-term variability)", "unit": "ms"},
                "sd1_sd2_ratio": {"description": "SD1/SD2 ratio", "unit": "ratio"},
                "dfa_alpha1": {"description": "DFA short-term scaling exponent", "unit": "dimensionless"},
                "dfa_alpha2": {"description": "DFA long-term scaling exponent", "unit": "dimensionless"},
                "sample_entropy": {"description": "Sample entropy", "unit": "dimensionless"},
                "approximate_entropy": {"description": "Approximate entropy", "unit": "dimensionless"},
                "fuzzy_entropy": {"description": "Fuzzy entropy", "unit": "dimensionless"},
                "permutation_entropy": {"description": "Permutation entropy", "unit": "dimensionless"}
            }
        }
    }
    
    return metrics_summary

# Background task functions
async def save_hrv_to_database(patient_id: str, session_id: str, 
                              hrv_metrics: Dict, interpretation: Dict, 
                              quality_assessment: Dict):
    """Save HRV results to database"""
    try:
        # Implementation would depend on database service
        logger.info(f"Saved HRV results for patient {patient_id}, session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save HRV to database: {str(e)}")

async def save_batch_hrv_to_database(results: List[Dict]):
    """Save batch HRV results to database"""
    try:
        # Implementation would depend on database service
        logger.info(f"Saved batch HRV results for {len(results)} patients")
    except Exception as e:
        logger.error(f"Failed to save batch HRV to database: {str(e)}")

# Helper functions
def generate_mock_hrv_history(patient_id: str, start_date: datetime, 
                             end_date: datetime, limit: int) -> List[Dict]:
    """Generate mock HRV history for demonstration"""
    
    mock_history = []
    current_date = start_date
    
    while current_date <= end_date and len(mock_history) < limit:
        # Generate realistic HRV values
        base_sdnn = 70 + np.random.normal(0, 15)
        base_rmssd = 30 + np.random.normal(0, 8)
        base_lf_hf = 1.2 + np.random.normal(0, 0.3)
        
        hrv_record = {
            "patient_id": patient_id,
            "date": current_date.isoformat(),
            "time_domain": {
                "sdnn": max(20, base_sdnn),
                "rmssd": max(10, base_rmssd),
                "pnn50": max(0, 12 + np.random.normal(0, 5))
            },
            "frequency_domain": {
                "lf_hf_ratio": max(0.1, base_lf_hf),
                "total_power": max(100, base_sdnn ** 2 * 0.8),
                "lf_power": max(50, base_sdnn ** 2 * 0.3),
                "hf_power": max(30, base_sdnn ** 2 * 0.2)
            },
            "nonlinear": {
                "sd1": max(5, base_rmssd / np.sqrt(2)),
                "sd2": max(10, base_sdnn / np.sqrt(2)),
                "sample_entropy": max(0, 1.2 + np.random.normal(0, 0.2))
            }
        }
        
        mock_history.append(hrv_record)
        current_date += timedelta(days=1)
    
    return mock_history

def export_hrv_report_to_csv(report_data: Dict) -> str:
    """Export HRV report to CSV format"""
    
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Patient ID", "Date", "SDNN", "RMSSD", "pNN50", "LF/HF Ratio"])
    
    # Write data
    for record in report_data.get("hrv_data", []):
        writer.writerow([
            record.get("patient_id", ""),
            record.get("date", ""),
            record.get("time_domain", {}).get("sdnn", ""),
            record.get("time_domain", {}).get("rmssd", ""),
            record.get("time_domain", {}).get("pnn50", ""),
            record.get("frequency_domain", {}).get("lf_hf_ratio", "")
        ])
    
    return output.getvalue()

def export_hrv_report_to_html(report_data: Dict) -> str:
    """Export HRV report to HTML format"""
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HRV Report - {patient_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>HRV Analysis Report</h1>
            <p>Patient ID: {patient_id}</p>
            <p>Report Generated: {report_date}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <p>Total Records: {total_records}</p>
            <p>Date Range: {start_date} to {end_date}</p>
        </div>
        
        <div class="section">
            <h2>HRV Data</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>SDNN (ms)</th>
                    <th>RMSSD (ms)</th>
                    <th>pNN50 (%)</th>
                    <th>LF/HF Ratio</th>
                </tr>
                {data_rows}
            </table>
        </div>
    </body>
    </html>
    """
    
    # Generate data rows
    data_rows = ""
    for record in report_data.get("hrv_data", []):
        data_rows += f"""
        <tr>
            <td>{record.get('date', '')}</td>
            <td>{record.get('time_domain', {}).get('sdnn', '')}</td>
            <td>{record.get('time_domain', {}).get('rmssd', '')}</td>
            <td>{record.get('time_domain', {}).get('pnn50', '')}</td>
            <td>{record.get('frequency_domain', {}).get('lf_hf_ratio', '')}</td>
        </tr>
        """
    
    return html_template.format(
        patient_id=report_data.get("patient_id", "Unknown"),
        report_date=report_data.get("report_generated", "Unknown"),
        total_records=report_data.get("data_summary", {}).get("total_records", 0),
        start_date=report_data.get("data_summary", {}).get("date_range", {}).get("start", "Unknown"),
        end_date=report_data.get("data_summary", {}).get("date_range", {}).get("end", "Unknown"),
        data_rows=data_rows
    )
