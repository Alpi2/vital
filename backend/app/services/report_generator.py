from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from datetime import datetime
import io

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='MedicalTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#0ea5e9'),
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=12
        ))
    
    def generate_patient_report(self, patient_data, anomalies, sessions):
        """Generate PDF report for a patient"""
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        story.append(Paragraph(
            f"ECG Analysis Report - {patient_data['first_name']} {patient_data['last_name']}",
            self.styles['MedicalTitle']
        ))
        
        # Patient Info
        story.append(self._create_patient_info_table(patient_data))
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))
        story.append(self._create_summary_table(sessions, anomalies))
        story.append(Spacer(1, 20))
        
        # Anomaly Details
        if anomalies:
            story.append(Paragraph("Detected Anomalies", self.styles['SectionHeader']))
            story.append(self._create_anomalies_table(anomalies))
            story.append(Spacer(1, 20))
        
        # BPM Trend Chart
        story.append(Paragraph("Heart Rate Trend", self.styles['SectionHeader']))
        story.append(self._create_bpm_chart(sessions))
        
        # Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            "This report is for informational purposes only. Consult a physician for medical advice.",
            self.styles['Italic']
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _create_patient_info_table(self, patient):
        data = [
            ["Patient ID", patient['medical_id']],
            ["Name", f"{patient['first_name']} {patient['last_name']}"],
            ["Date of Birth", patient['date_of_birth']],
            ["Age", self._calculate_age(patient['date_of_birth'])],
            ["Gender", patient.get('gender', 'Not specified')],
            ["Blood Type", patient.get('blood_type', 'Unknown')]
        ]
        
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        return table
    
    def _create_summary_table(self, sessions, anomalies):
        total_sessions = len(sessions)
        total_anomalies = len(anomalies)
        avg_bpm = sum(s['average_bpm'] for s in sessions) / total_sessions if sessions else 0
        
        data = [
            ["Metric", "Value"],
            ["Total Monitoring Sessions", str(total_sessions)],
            ["Total Anomalies Detected", str(total_anomalies)],
            ["Average Heart Rate", f"{avg_bpm:.1f} BPM"],
            ["Monitoring Period", f"{sessions[0]['created_at'].date()} to {sessions[-1]['created_at'].date()}" if sessions else "N/A"]
        ]
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0ea5e9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _calculate_age(self, dob):
        if isinstance(dob, str):
            dob = datetime.strptime(dob, '%Y-%m-%d')
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

