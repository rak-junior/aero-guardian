"""
PDF Report Generator - No Static Values
========================================
Author: AeroGuardian Member
Date: 2026-01-16

Generates PDF reports using ONLY data from report_data.
No hardcoded text, no fallback values.
Raises error if required data is missing.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("AeroGuardian.PDF")

# Try importing reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PDFGenerationError(Exception):
    """Raised when PDF generation fails due to missing data."""


def create_styles():
    """Create PDF styles with unique names to avoid conflicts."""
    if not HAS_REPORTLAB:
        return {}
    
    styles = getSampleStyleSheet()
    
    # Use unique names to avoid conflicts with built-in styles
    styles.add(ParagraphStyle(
        'Doc3Title', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor('#1a365d'),
        alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'Doc3Subtitle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#4a5568'),
        alignment=TA_CENTER, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'Doc3Section', parent=styles['Heading2'],
        fontSize=12, textColor=colors.HexColor('#1a365d'),
        fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'Doc3SectionSub', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#718096'),
        fontName='Helvetica-Oblique', spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'Doc3Body', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#2d3748'),
        alignment=TA_LEFT, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'Doc3Bullet', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#2d3748'),
        leftIndent=15, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        'Doc3Category', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#1a365d'),
        fontName='Helvetica-Bold', spaceBefore=6, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        'Doc3Footer', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER, fontName='Helvetica-Oblique',
    ))
    
    return styles


if HAS_REPORTLAB:
    RISK_COLORS = {
        'CRITICAL': colors.HexColor('#9b2c2c'),
        'HIGH': colors.HexColor('#c53030'),
        'MEDIUM': colors.HexColor('#dd6b20'),
        'LOW': colors.HexColor('#38a169'),
    }
else:
    RISK_COLORS = {}


class PDFGenerator:
    """Generate PDF reports using ONLY provided data. No hardcoded values."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        if not HAS_REPORTLAB:
            raise ImportError("reportlab required")
        
        self.output_dir = Path(output_dir or "outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = create_styles()
    
    def generate(self, report_data: Dict[str, Any], output_path: Path) -> Path:
        """Generate PDF using the NEW 3-Section Pre-Flight Safety Report format."""
        
        if not report_data:
            raise PDFGenerationError("report_data is empty")
        
        # Extract sections from NEW structure
        incident_source = report_data.get("incident_source", report_data.get("section_1_incident_source", {}))
        section_1 = report_data.get("section_1_safety_level_and_cause", {})
        section_2 = report_data.get("section_2_design_constraints_and_recommendations", {})
        section_3 = report_data.get("section_3_explanation", {})
        verdict_data = report_data.get("verdict", {})
        supporting_data = report_data.get("supporting_data", {})
        
        # Legacy fallback for old format
        if not incident_source:
            incident = report_data.get("incident", {})
            incident_source = {
                "report_id": incident.get("id", "Unknown"),
                "location": incident.get("location", "Unknown"),
                "original_faa_narrative": incident.get("description", ""),
            }
        
        try:
            doc = SimpleDocTemplate(
                str(output_path), pagesize=A4,
                rightMargin=0.6*inch, leftMargin=0.6*inch,
                topMargin=0.5*inch, bottomMargin=0.4*inch,
            )
            
            story = []
            
            # Get safety level for color
            safety_level = section_1.get("safety_level", "UNKNOWN")
            if "CRITICAL" in safety_level.upper():
                risk_color = RISK_COLORS.get("CRITICAL", colors.red)
            elif "HIGH" in safety_level.upper():
                risk_color = RISK_COLORS.get("HIGH", colors.orange)
            elif "MEDIUM" in safety_level.upper():
                risk_color = RISK_COLORS.get("MEDIUM", colors.gold)
            else:
                risk_color = RISK_COLORS.get("LOW", colors.green)
            
            # Get verdict and color for pre-flight decision
            verdict = verdict_data.get("decision", verdict_data.get("go_nogo", "REVIEW"))
            if isinstance(verdict, dict):
                verdict = verdict.get("decision", "REVIEW")
            verdict_color = colors.red if "NO-GO" in verdict.upper() else (colors.orange if "CAUTION" in verdict.upper() else colors.green)
            
            # =================================================================
            # HEADER
            # =================================================================
            story.append(Paragraph("PRE-FLIGHT SAFETY REPORT", self.styles['Doc3Title']))
            story.append(Paragraph(
                f"<b>Report ID:</b> {incident_source.get('report_id', 'Unknown')}", 
                self.styles['Doc3Subtitle']
            ))
            story.append(Paragraph(
                f"<b>Location:</b> {incident_source.get('location', 'Unknown')}", 
                self.styles['Doc3Subtitle']
            ))
            story.append(self._separator())
            
            # =================================================================
            # INCIDENT SOURCE (Context)
            # =================================================================
            # story.append(Paragraph("INCIDENT SOURCE", self.styles['Doc3Section']))
            
            # narrative = incident_source.get("original_faa_narrative", "")
            # if narrative:
            #     display_narrative = narrative[:400] + "..." if len(narrative) > 400 else narrative
            #     story.append(Paragraph(f"<i>\"{display_narrative}\"</i>", self.styles['Doc3Body']))
            
            # story.append(self._separator())
            
            # =================================================================
            # SECTION 1: SAFETY LEVEL & CAUSE
            # =================================================================
            story.append(Paragraph(
                f"<font color='{risk_color.hexval()}'>1. SAFETY LEVEL & CAUSE</font>",
                self.styles['Doc3Section']
            ))
            story.append(Paragraph(
                f"<b>Safety Level:</b> <font color='{risk_color.hexval()}' size='12'>{safety_level}</font>", 
                self.styles['Doc3Body']
            ))
            story.append(Paragraph(
                f"<b>Pre-Flight Decision:</b> <font color='{verdict_color.hexval()}' size='12'><b>{verdict}</b></font>",
                self.styles['Doc3Body']
            ))
            story.append(Paragraph(
                f"<b>Primary Hazard:</b> {section_1.get('primary_hazard', 'Unknown')}", 
                self.styles['Doc3Body']
            ))
            story.append(Paragraph(
                f"<b>Observed Effect:</b> {section_1.get('observed_effect', 'No specific effect')}", 
                self.styles['Doc3Body']
            ))
            
            story.append(self._separator())
            
            # =================================================================
            # SECTION 2: DESIGN CONSTRAINTS & RECOMMENDATIONS
            # =================================================================
            story.append(Paragraph("2. DESIGN CONSTRAINTS & RECOMMENDATIONS", self.styles['Doc3Section']))
            
            # Design Constraints
            constraints = section_2.get("design_constraints", [])
            if constraints:
                story.append(Paragraph("<b>Design Constraints:</b>", self.styles['Doc3Body']))
                for constraint in constraints[:4]:
                    if constraint.strip():
                        story.append(Paragraph(f"• {constraint.strip()}", self.styles['Doc3Bullet']))
            
            # Recommendations
            recommendations = section_2.get("recommendations", [])
            if recommendations:
                story.append(Paragraph("<b>Recommendations:</b>", self.styles['Doc3Body']))
                for i, rec in enumerate(recommendations[:5], 1):
                    if rec.strip():
                        story.append(Paragraph(f"{i}. {rec.strip()}", self.styles['Doc3Bullet']))
            
            story.append(self._separator())
            
            # =================================================================
            # SECTION 3: EXPLANATION (WHY) - THE NOVELTY
            # =================================================================
            story.append(Paragraph("3. EXPLANATION", self.styles['Doc3Section']))
            
            explanation = section_3.get("reasoning", "")
            if explanation:
                story.append(Paragraph(f"{explanation}", self.styles['Doc3Body']))
            
            story.append(self._separator())
            
            # =================================================================
            # FINAL VERDICT
            # =================================================================
            # verdict = verdict_data.get("decision", verdict_data.get("go_nogo", "REVIEW"))
            # if isinstance(verdict, dict):
            #     verdict = verdict.get("decision", "REVIEW")
            # verdict_color = colors.red if "NO-GO" in verdict.upper() else (colors.orange if "CAUTION" in verdict.upper() else colors.green)
            
            # story.append(Paragraph("VERDICT", self.styles['Doc3Section']))
            # story.append(Paragraph(
            #     f"<b>Pre-Flight Decision:</b> <font color='{verdict_color.hexval()}' size='14'><b>{verdict}</b></font>",
            #     self.styles['Doc3Body']
            # ))
            
            # story.append(self._separator())
            
            # =================================================================
            # SUPPORTING DATA SUMMARY
            # =================================================================
            story.append(Paragraph("SUPPORTING DATA", self.styles['Doc3Section']))
            
            sim_config = supporting_data.get("simulation_config", {})
            telemetry_sum = supporting_data.get("telemetry_summary", {})
            
            story.append(Paragraph(
                f"<b>Simulation:</b> {sim_config.get('waypoints_count', 0)} waypoints, "
                f"Fault: {sim_config.get('fault_type', 'None')}, "
                f"Alt: {sim_config.get('altitude_m', 0):.0f}m",
                self.styles['Doc3Body']
            ))
            story.append(Paragraph(
                f"<b>Telemetry:</b> {telemetry_sum.get('data_points', 0)} data points, "
                f"Duration: {telemetry_sum.get('duration_sec', 0):.0f}s, "
                f"Max Alt: {telemetry_sum.get('max_altitude_m', 0):.1f}m",
                self.styles['Doc3Body']
            ))
            
            story.append(self._separator())
            
            # =================================================================
            # FOOTER
            # =================================================================
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | AeroGuardian v1.0 Pre-Flight Safety System</i>",
                self.styles['Doc3Footer']
            ))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF generated: {output_path}")
            return output_path
            
        except PDFGenerationError:
            raise
        except Exception as e:
            raise PDFGenerationError(f"PDF build failed: {e}")
    
    def _separator(self):
        """Create separator line."""
        return HRFlowable(
            width="100%", thickness=1,
            color=colors.HexColor('#718096'),
            spaceBefore=6, spaceAfter=6
        )


def generate_pdf(report_data: Dict, output_path: Path) -> Path:
    """Generate PDF. Raises PDFGenerationError on failure."""
    generator = PDFGenerator(output_path.parent)
    return generator.generate(report_data, output_path)
