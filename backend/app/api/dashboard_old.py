"""Dashboard API endpoints for admin panel."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/kpis")
async def get_kpis() -> Dict[str, Any]:
    """Get KPI metrics for executive dashboard."""
    return {
        "operational": [
            {
                "id": "bed_occupancy",
                "name": "Yatak Doluluk Oranı",
                "value": 87.5,
                "target": 85.0,
                "unit": "%",
                "trend": "up",
                "change": 2.3,
                "category": "operational"
            },
            {
                "id": "avg_los",
                "name": "Ortalama Kalış Süresi",
                "value": 4.2,
                "target": 4.5,
                "unit": "gün",
                "trend": "down",
                "change": -0.3,
                "category": "operational"
            },
            {
                "id": "er_wait_time",
                "name": "Acil Servis Bekleme Süresi",
                "value": 23,
                "target": 30,
                "unit": "dakika",
                "trend": "down",
                "change": -5,
                "category": "operational"
            }
        ],
        "quality": [
            {
                "id": "patient_satisfaction",
                "name": "Hasta Memnuniyeti",
                "value": 92.3,
                "target": 90.0,
                "unit": "%",
                "trend": "up",
                "change": 1.5,
                "category": "quality"
            },
            {
                "id": "readmission_rate",
                "name": "Yeniden Yatış Oranı",
                "value": 8.7,
                "target": 10.0,
                "unit": "%",
                "trend": "down",
                "change": -1.2,
                "category": "quality"
            }
        ],
        "financial": [
            {
                "id": "revenue_per_patient",
                "name": "Hasta Başına Gelir",
                "value": 15420,
                "target": 15000,
                "unit": "TL",
                "trend": "up",
                "change": 2.8,
                "category": "financial"
            },
            {
                "id": "cost_per_patient",
                "name": "Hasta Başına Maliyet",
                "value": 12350,
                "target": 13000,
                "unit": "TL",
                "trend": "down",
                "change": -5.0,
                "category": "financial"
            }
        ],
        "clinical": [
            {
                "id": "mortality_rate",
                "name": "Ölüm Oranı",
                "value": 1.2,
                "target": 1.5,
                "unit": "%",
                "trend": "down",
                "change": -0.2,
                "category": "clinical"
            },
            {
                "id": "infection_rate",
                "name": "Enfeksiyon Oranı",
                "value": 2.1,
                "target": 2.5,
                "unit": "%",
                "trend": "down",
                "change": -0.3,
                "category": "clinical"
            }
        ],
        "hr": [
            {
                "id": "staff_turnover",
                "name": "Personel Devir Oranı",
                "value": 12.5,
                "target": 15.0,
                "unit": "%",
                "trend": "down",
                "change": -1.8,
                "category": "hr"
            },
            {
                "id": "nurse_patient_ratio",
                "name": "Hemşire/Hasta Oranı",
                "value": 1.5,
                "target": 1.3,
                "unit": "ratio",
                "trend": "stable",
                "change": 0.0,
                "category": "hr"
            }
        ]
    }


@router.get("/realtime/vitals")
async def get_realtime_vitals() -> Dict[str, Any]:
    """Get real-time system vitals."""
    return {
        "timestamp": datetime.now().isoformat(),
        "active_patients": 142,
        "active_monitors": 138,
        "active_alarms": 7,
        "critical_alarms": 2,
        "system_health": {
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "disk_usage": 38.5,
            "network_latency": 12.3
        },
        "departments": [
            {"name": "ICU", "patients": 24, "capacity": 30, "utilization": 80.0},
            {"name": "ER", "patients": 18, "capacity": 20, "utilization": 90.0},
            {"name": "Cardiology", "patients": 32, "capacity": 40, "utilization": 80.0},
            {"name": "Pediatrics", "patients": 28, "capacity": 35, "utilization": 80.0},
            {"name": "Surgery", "patients": 40, "capacity": 50, "utilization": 80.0}
        ]
    }


@router.get("/utilization")
async def get_utilization_reports() -> Dict[str, Any]:
    """Get utilization reports for beds, devices, staff, and rooms."""
    return {
        "beds": [
            {"department": "ICU", "total": 30, "occupied": 24, "available": 6, "utilization": 80.0, "status": "normal"},
            {"department": "ER", "total": 20, "occupied": 18, "available": 2, "utilization": 90.0, "status": "high"},
            {"department": "Cardiology", "total": 40, "occupied": 32, "available": 8, "utilization": 80.0, "status": "normal"},
            {"department": "Pediatrics", "total": 35, "occupied": 28, "available": 7, "utilization": 80.0, "status": "normal"},
            {"department": "Surgery", "total": 50, "occupied": 40, "available": 10, "utilization": 80.0, "status": "normal"}
        ],
        "devices": [
            {"type": "Vital Signs Monitor", "total": 150, "in_use": 138, "available": 12, "maintenance": 0, "utilization": 92.0},
            {"type": "Ventilator", "total": 45, "in_use": 28, "available": 15, "maintenance": 2, "utilization": 62.2},
            {"type": "Infusion Pump", "total": 200, "in_use": 165, "available": 30, "maintenance": 5, "utilization": 82.5},
            {"type": "ECG Machine", "total": 80, "in_use": 52, "available": 25, "maintenance": 3, "utilization": 65.0},
            {"type": "Defibrillator", "total": 60, "in_use": 12, "available": 46, "maintenance": 2, "utilization": 20.0}
        ],
        "staff": [
            {"role": "Nurse", "scheduled": 120, "present": 115, "absent": 5, "utilization": 95.8, "shift": "Day"},
            {"role": "Doctor", "scheduled": 45, "present": 43, "absent": 2, "utilization": 95.6, "shift": "Day"},
            {"role": "Technician", "scheduled": 30, "present": 28, "absent": 2, "utilization": 93.3, "shift": "Day"},
            {"role": "Support Staff", "scheduled": 50, "present": 48, "absent": 2, "utilization": 96.0, "shift": "Day"}
        ],
        "rooms": [
            {"type": "Operating Room", "total": 12, "in_use": 8, "available": 3, "cleaning": 1, "utilization": 66.7},
            {"type": "ICU Room", "total": 30, "in_use": 24, "available": 6, "cleaning": 0, "utilization": 80.0},
            {"type": "ER Bay", "total": 20, "in_use": 18, "available": 2, "cleaning": 0, "utilization": 90.0},
            {"type": "Exam Room", "total": 45, "in_use": 32, "available": 10, "cleaning": 3, "utilization": 71.1}
        ]
    }


@router.get("/quality-metrics")
async def get_quality_metrics() -> Dict[str, Any]:
    """Get quality metrics dashboard data."""
    return {
        "metrics": [
            {"id": "patient_safety", "name": "Hasta Güvenliği Skoru", "score": 94.5, "benchmark": 90.0, "category": "safety"},
            {"id": "infection_control", "name": "Enfeksiyon Kontrolü", "score": 96.2, "benchmark": 95.0, "category": "safety"},
            {"id": "medication_safety", "name": "İlaç Güvenliği", "score": 98.1, "benchmark": 97.0, "category": "safety"},
            {"id": "fall_prevention", "name": "Düşme Önleme", "score": 92.8, "benchmark": 90.0, "category": "safety"},
            {"id": "pressure_ulcer", "name": "Basınç Yarası Önleme", "score": 95.3, "benchmark": 93.0, "category": "safety"},
            {"id": "patient_satisfaction", "name": "Hasta Memnuniyeti", "score": 92.3, "benchmark": 88.0, "category": "experience"},
            {"id": "staff_satisfaction", "name": "Personel Memnuniyeti", "score": 87.5, "benchmark": 85.0, "category": "experience"},
            {"id": "clinical_outcomes", "name": "Klinik Sonuçlar", "score": 93.7, "benchmark": 90.0, "category": "outcomes"},
            {"id": "readmission_rate", "name": "Yeniden Yatış Oranı", "score": 91.3, "benchmark": 90.0, "category": "outcomes"},
            {"id": "mortality_rate", "name": "Ölüm Oranı", "score": 98.8, "benchmark": 98.5, "category": "outcomes"}
        ],
        "trends": [
            {"month": "Ocak", "safety": 93.2, "experience": 89.5, "outcomes": 92.1},
            {"month": "Şubat", "safety": 94.1, "experience": 90.2, "outcomes": 92.8},
            {"month": "Mart", "safety": 94.8, "experience": 91.0, "outcomes": 93.5},
            {"month": "Nisan", "safety": 95.2, "experience": 91.5, "outcomes": 94.0},
            {"month": "Mayıs", "safety": 95.5, "experience": 92.0, "outcomes": 94.3},
            {"month": "Haziran", "safety": 95.8, "experience": 92.3, "outcomes": 94.6}
        ]
    }


@router.get("/financial-analytics")
async def get_financial_analytics() -> Dict[str, Any]:
    """Get financial analytics dashboard data."""
    return {
        "summary": {
            "total_revenue": 12500000,
            "total_expenses": 9800000,
            "net_profit": 2700000,
            "profit_margin": 21.6,
            "roi": 27.6,
            "ebitda": 3200000
        },
        "revenue_by_department": [
            {"department": "Cardiology", "revenue": 3200000, "percentage": 25.6},
            {"department": "Surgery", "revenue": 2800000, "percentage": 22.4},
            {"department": "ICU", "revenue": 2100000, "percentage": 16.8},
            {"department": "ER", "revenue": 1800000, "percentage": 14.4},
            {"department": "Pediatrics", "revenue": 1400000, "percentage": 11.2},
            {"department": "Other", "revenue": 1200000, "percentage": 9.6}
        ],
        "expenses_by_category": [
            {"category": "Personnel", "amount": 5400000, "percentage": 55.1},
            {"category": "Medical Supplies", "amount": 2200000, "percentage": 22.4},
            {"category": "Equipment", "amount": 1100000, "percentage": 11.2},
            {"category": "Utilities", "amount": 600000, "percentage": 6.1},
            {"category": "Other", "amount": 500000, "percentage": 5.1}
        ],
        "cash_flow": [
            {"month": "Ocak", "inflow": 2100000, "outflow": 1650000, "net": 450000},
            {"month": "Şubat", "inflow": 2050000, "outflow": 1620000, "net": 430000},
            {"month": "Mart", "inflow": 2150000, "outflow": 1680000, "net": 470000},
            {"month": "Nisan", "inflow": 2080000, "outflow": 1640000, "net": 440000},
            {"month": "Mayıs", "inflow": 2120000, "outflow": 1660000, "net": 460000},
            {"month": "Haziran", "inflow": 2000000, "outflow": 1550000, "net": 450000}
        ],
        "profitability": {
            "gross_margin": 28.5,
            "operating_margin": 23.2,
            "net_margin": 21.6,
            "return_on_assets": 18.3,
            "return_on_equity": 27.6
        }
    }


@router.get("/predictive-maintenance")
async def get_predictive_maintenance() -> Dict[str, Any]:
    """Get predictive maintenance dashboard data."""
    return {
        "devices": [
            {
                "id": "VM-001",
                "name": "Vital Signs Monitor #1",
                "type": "Monitor",
                "health_score": 92,
                "predicted_failure_date": (datetime.now() + timedelta(days=180)).isoformat(),
                "last_maintenance": (datetime.now() - timedelta(days=45)).isoformat(),
                "next_maintenance": (datetime.now() + timedelta(days=45)).isoformat(),
                "status": "healthy"
            },
            {
                "id": "VENT-012",
                "name": "Ventilator #12",
                "type": "Ventilator",
                "health_score": 78,
                "predicted_failure_date": (datetime.now() + timedelta(days=60)).isoformat(),
                "last_maintenance": (datetime.now() - timedelta(days=75)).isoformat(),
                "next_maintenance": (datetime.now() + timedelta(days=15)).isoformat(),
                "status": "warning"
            },
            {
                "id": "PUMP-045",
                "name": "Infusion Pump #45",
                "type": "Pump",
                "health_score": 65,
                "predicted_failure_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "last_maintenance": (datetime.now() - timedelta(days=90)).isoformat(),
                "next_maintenance": (datetime.now() + timedelta(days=7)).isoformat(),
                "status": "critical"
            },
            {
                "id": "ECG-023",
                "name": "ECG Machine #23",
                "type": "ECG",
                "health_score": 88,
                "predicted_failure_date": (datetime.now() + timedelta(days=150)).isoformat(),
                "last_maintenance": (datetime.now() - timedelta(days=30)).isoformat(),
                "next_maintenance": (datetime.now() + timedelta(days=60)).isoformat(),
                "status": "healthy"
            }
        ],
        "alerts": [
            {
                "id": "alert-001",
                "device_id": "PUMP-045",
                "severity": "critical",
                "message": "Infusion Pump #45 requires immediate maintenance",
                "predicted_failure": "30 days",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "alert-002",
                "device_id": "VENT-012",
                "severity": "warning",
                "message": "Ventilator #12 maintenance due in 15 days",
                "predicted_failure": "60 days",
                "created_at": datetime.now().isoformat()
            }
        ],
        "maintenance_schedule": [
            {
                "date": (datetime.now() + timedelta(days=7)).isoformat(),
                "device_id": "PUMP-045",
                "device_name": "Infusion Pump #45",
                "type": "preventive",
                "priority": "critical"
            },
            {
                "date": (datetime.now() + timedelta(days=15)).isoformat(),
                "device_id": "VENT-012",
                "device_name": "Ventilator #12",
                "type": "preventive",
                "priority": "high"
            },
            {
                "date": (datetime.now() + timedelta(days=45)).isoformat(),
                "device_id": "VM-001",
                "device_name": "Vital Signs Monitor #1",
                "type": "routine",
                "priority": "normal"
            }
        ]
    }


@router.get("/compliance-reports")
async def get_compliance_reports() -> Dict[str, Any]:
    """Get compliance reports dashboard data."""
    return {
        "reports": [
            {
                "id": "hipaa",
                "name": "HIPAA Compliance",
                "category": "privacy",
                "compliance_rate": 98.5,
                "last_audit": (datetime.now() - timedelta(days=30)).isoformat(),
                "next_audit": (datetime.now() + timedelta(days=60)).isoformat(),
                "status": "compliant",
                "findings": 2,
                "critical_findings": 0
            },
            {
                "id": "gdpr",
                "name": "GDPR/KVKK Compliance",
                "category": "privacy",
                "compliance_rate": 97.2,
                "last_audit": (datetime.now() - timedelta(days=45)).isoformat(),
                "next_audit": (datetime.now() + timedelta(days=45)).isoformat(),
                "status": "compliant",
                "findings": 3,
                "critical_findings": 0
            },
            {
                "id": "fda",
                "name": "FDA Regulations",
                "category": "medical_device",
                "compliance_rate": 99.1,
                "last_audit": (datetime.now() - timedelta(days=60)).isoformat(),
                "next_audit": (datetime.now() + timedelta(days=30)).isoformat(),
                "status": "compliant",
                "findings": 1,
                "critical_findings": 0
            },
            {
                "id": "jci",
                "name": "JCI Accreditation",
                "category": "quality",
                "compliance_rate": 96.8,
                "last_audit": (datetime.now() - timedelta(days=90)).isoformat(),
                "next_audit": (datetime.now() + timedelta(days=270)).isoformat(),
                "status": "compliant",
                "findings": 5,
                "critical_findings": 1
            },
            {
                "id": "iso",
                "name": "ISO 13485",
                "category": "quality",
                "compliance_rate": 98.0,
                "last_audit": (datetime.now() - timedelta(days=120)).isoformat(),
                "next_audit": (datetime.now() + timedelta(days=240)).isoformat(),
                "status": "compliant",
                "findings": 2,
                "critical_findings": 0
            }
        ],
        "findings": [
            {
                "id": "finding-001",
                "report_id": "jci",
                "severity": "critical",
                "category": "patient_safety",
                "description": "Incomplete documentation in 2 patient records",
                "status": "open",
                "due_date": (datetime.now() + timedelta(days=14)).isoformat()
            },
            {
                "id": "finding-002",
                "report_id": "gdpr",
                "severity": "medium",
                "category": "data_privacy",
                "description": "Missing consent forms for 3 patients",
                "status": "in_progress",
                "due_date": (datetime.now() + timedelta(days=21)).isoformat()
            }
        ]
    }
