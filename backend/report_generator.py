"""
Report Generation Module for CircuitCheck
Generates PDF reports and handles data export functionality
"""
# pyright: reportMissingImports=false

import os
import io
import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, cast
import logging

# Define fallback classes and constants first
from types import SimpleNamespace

# Define fallback constants
A4_FALLBACK = (595.27, 841.89)  # A4 size in points
INCH_FALLBACK = 72
TA_CENTER_FALLBACK = 1
TA_LEFT_FALLBACK = 0 
TA_RIGHT_FALLBACK = 2

# Create fallback colors module
colors_fallback = SimpleNamespace(
    HexColor=lambda x: x,
    black='#000000',
    white='#FFFFFF'
)

# Create dummy classes for fallback
def getSampleStyleSheet_fallback():
    """Return a mapping whose values look like real ParagraphStyle objects.

    Returning plain dicts caused Pylance to widen the type of styles['X'] to
    dict[str, Unknown] | ParagraphStyle which then produced errors when used
    as the 'parent' argument for real ReportLab ParagraphStyle. We instead
    return ParagraphStyle_fallback instances so every style access yields a
    uniform object with attribute-style access resembling the real API.
    """
    return {
        'Title': ParagraphStyle_fallback('Title', fontSize=16, textColor='#000000'),
        'Normal': ParagraphStyle_fallback('Normal', fontSize=10, textColor='#000000'),
        'Heading2': ParagraphStyle_fallback('Heading2', fontSize=14, textColor='#000000'),
        'Caption': ParagraphStyle_fallback('Caption', fontSize=8, textColor='#000000')
    }

class ParagraphStyle_fallback:
    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

class SimpleDocTemplate_fallback:
    def __init__(self, filename, **kwargs):
        self.filename = filename
    def build(self, story):
        # Fallback: just create an empty file
        with open(self.filename, 'w') as f:
            f.write("PDF generation requires ReportLab library\n")

class Paragraph_fallback:
    def __init__(self, text, style=None):
        self.text = text
        self.style = style

class Spacer_fallback:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Table_fallback:
    def __init__(self, data, colWidths=None, **kwargs):
        self.data = data
        self.colWidths = colWidths
    def setStyle(self, style):
        pass

class TableStyle_fallback:
    def __init__(self, commands):
        self.commands = commands
    
    def __iter__(self):
        return iter(self.commands)

class Image_fallback:
    def __init__(self, filename, width=None, height=None, **kwargs):
        self.filename = filename
        self.width = width
        self.height = height

class PageBreak_fallback:
    def __init__(self):
        pass

# Optional imports - handle gracefully if not available
REPORTLAB_AVAILABLE = False
try:
    # Optional dependency: reportlab (silence static analysis if not installed)
    import reportlab.lib.colors as rl_colors  # type: ignore[import-not-found]
    from reportlab.lib.pagesizes import letter, A4 as RL_A4  # type: ignore[import-not-found]
    from reportlab.lib.styles import getSampleStyleSheet as rl_getSampleStyleSheet, ParagraphStyle as RL_ParagraphStyle  # type: ignore[import-not-found]
    from reportlab.lib.units import inch as rl_inch  # type: ignore[import-not-found]
    from reportlab.platypus import SimpleDocTemplate as RL_SimpleDocTemplate, Paragraph as RL_Paragraph, Spacer as RL_Spacer, Table as RL_Table, TableStyle as RL_TableStyle, Image as RL_Image  # type: ignore[import-not-found]
    from reportlab.platypus.flowables import PageBreak as RL_PageBreak  # type: ignore[import-not-found]
    from reportlab.graphics.shapes import Drawing  # type: ignore[import-not-found]
    from reportlab.graphics.charts.piecharts import Pie  # type: ignore[import-not-found]
    from reportlab.graphics.charts.barcharts import VerticalBarChart  # type: ignore[import-not-found]
    from reportlab.lib.enums import TA_CENTER as RL_TA_CENTER, TA_LEFT as RL_TA_LEFT, TA_RIGHT as RL_TA_RIGHT  # type: ignore[import-not-found]
    
    # Assign ReportLab classes to the names used in the code
    colors = rl_colors
    A4 = RL_A4
    getSampleStyleSheet = rl_getSampleStyleSheet
    ParagraphStyle = RL_ParagraphStyle
    inch = rl_inch
    SimpleDocTemplate = RL_SimpleDocTemplate
    Paragraph = RL_Paragraph
    Spacer = RL_Spacer
    Table = RL_Table
    TableStyle = RL_TableStyle
    Image = RL_Image
    PageBreak = RL_PageBreak
    TA_CENTER = RL_TA_CENTER
    TA_LEFT = RL_TA_LEFT
    TA_RIGHT = RL_TA_RIGHT
    
    REPORTLAB_AVAILABLE = True
except ImportError:
    # Use fallback implementations
    colors = colors_fallback
    A4 = A4_FALLBACK
    getSampleStyleSheet = getSampleStyleSheet_fallback
    ParagraphStyle = ParagraphStyle_fallback
    inch = INCH_FALLBACK
    SimpleDocTemplate = SimpleDocTemplate_fallback
    Paragraph = Paragraph_fallback
    Spacer = Spacer_fallback
    Table = Table_fallback
    TableStyle = TableStyle_fallback
    Image = Image_fallback
    PageBreak = PageBreak_fallback
    TA_CENTER = TA_CENTER_FALLBACK
    TA_LEFT = TA_LEFT_FALLBACK
    TA_RIGHT = TA_RIGHT_FALLBACK
    
    logging.warning("ReportLab not available. PDF generation will use fallback implementation.")

# Helper functions to get the right values based on availability  
# (Avoid runtime imports that trigger missing-import diagnostics when ReportLab is absent.)
def get_page_size():
    # A4 is already bound either to real ReportLab A4 or fallback tuple.
    return A4

def get_inch():
    # inch is already bound either to real ReportLab unit or fallback value.
    return inch

def get_ta_center():
    return TA_CENTER

def get_ta_left():
    return TA_LEFT

def get_ta_right():
    return TA_RIGHT

try:
    import base64
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow not available. Image processing will be limited.")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Typing helpers (minimise Problems panel noise) -----------------
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Only for static analysis; at runtime these may not exist
    try:  # pragma: no cover - purely a typing aid
        from reportlab.lib.styles import ParagraphStyle as RLParagraphStyle  # type: ignore
    except Exception:  # Fallback stub
        class RLParagraphStyle:  # type: ignore
            ...
else:  # Runtime alias (we don't care – we use duck typing)
    try:
        from reportlab.lib.styles import ParagraphStyle as RLParagraphStyle  # type: ignore
    except Exception:
        class RLParagraphStyle:  # type: ignore
            ...

def _resolve_parent_style(candidate: Any):
    """Return candidate only if it is a real ReportLab ParagraphStyle.

    This guards the 'parent=' parameter of ParagraphStyle creation which expects
    a PropertySet/ParagraphStyle. Our fallback objects are lightweight and should
    not be passed as parent to avoid static type errors and potential runtime
    incompatibilities.
    """
    try:
        if REPORTLAB_AVAILABLE and isinstance(candidate, RLParagraphStyle):
            return candidate
    except Exception:
        pass
    return None

def _safe_get_style(styles_map: Any, key: str):
    """Retrieve a style by key from either a StyleSheet1 or our fallback dict.

    Always returns the raw object; parent suitability is handled separately.
    """
    try:
        if hasattr(styles_map, 'get'):  # StyleSheet1 has .get
            return styles_map.get(key)  # type: ignore[no-any-return]
        return styles_map[key]
    except Exception:
        return None

def _make_paragraph_style(name: str, base_key: str, styles_map: Any, **overrides: Any):
    """Create a ParagraphStyle with a validated parent.

    If the underlying style can't be used as a parent we simply pass parent=None.
    This avoids the frequent Problems panel error:
        Argument of type "PropertySet | dict[str, Unknown] | None" ...

    We branch so the 'parent' argument is only provided when not None, preventing
    Pyright from widening the union (RLParagraphStyle | None) for the call; a cast
    is used to silence the mismatch between our lightweight runtime stub and the
    real ReportLab PropertySet type.
    """
    base_raw = _safe_get_style(styles_map, base_key)
    parent = _resolve_parent_style(base_raw)
    if parent is not None:
        return ParagraphStyle(name, parent=cast(Any, parent), **overrides)
    return ParagraphStyle(name, **overrides)


def _make_color(value: str):
    """Return a color object or raw value depending on environment.

    This prevents static analysis from widening types when colors.HexColor
    is not available; we just return the string in fallback.
    """
    try:
        if REPORTLAB_AVAILABLE and hasattr(colors, 'HexColor'):
            return colors.HexColor(value)  # type: ignore[no-any-return]
    except Exception:
        pass
    return value


class ReportGenerator:
    """Main class for generating analysis reports in various formats"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'reports')
        self.ensure_output_directory()
        
        # Report styling configuration
        self.report_config = {
            'page_size': get_page_size(),
            'margins': {
                'top': 72,
                'bottom': 72,
                'left': 72,
                'right': 72
            },
            'colors': {
                'primary': _make_color('#1976d2'),
                'secondary': _make_color('#424242'),
                'success': _make_color('#4caf50'),
                'warning': _make_color('#ff9800'),
                'error': _make_color('#f44336'),
                'background': _make_color('#f5f5f5')
            },
            'fonts': {
                'title': 16,
                'heading': 14,
                'body': 10,
                'caption': 8
            }
        }
    
    def ensure_output_directory(self):
        """Ensure the output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_analysis_report(
        self, 
        test_result: Dict[str, Any], 
        format: str = 'pdf',
        include_images: bool = True,
        include_raw_data: bool = False
    ) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            test_result: Test result data
            format: Output format ('pdf', 'html', 'json')
            include_images: Whether to include component images
            include_raw_data: Whether to include raw measurement data
            
        Returns:
            Path to generated report file
        """
        
        if format.lower() == 'pdf':
            return self._generate_pdf_report(test_result, include_images, include_raw_data)
        elif format.lower() == 'html':
            return self._generate_html_report(test_result, include_images, include_raw_data)
        elif format.lower() == 'json':
            return self._generate_json_report(test_result, include_raw_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_pdf_report(
        self, 
        test_result: Dict[str, Any], 
        include_images: bool = True,
        include_raw_data: bool = False
    ) -> str:
        """Generate PDF report using ReportLab"""
        
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback_report(test_result, 'pdf')
        
        # Generate filename
        test_id = test_result.get('test_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{test_id}_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=self.report_config['page_size'],
            topMargin=self.report_config['margins']['top'],
            bottomMargin=self.report_config['margins']['bottom'],
            leftMargin=self.report_config['margins']['left'],
            rightMargin=self.report_config['margins']['right']
        )
        
        # Build story (content)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles created via helper to avoid parent type noise in static analysis
        title_style = _make_paragraph_style(  # type: ignore[call-arg] - fallback parent union not representable
            'CustomTitle',
            'Title',
            styles,
            fontSize=self.report_config['fonts']['title'],
            textColor=self.report_config['colors']['primary'],
            alignment=get_ta_center(),
            spaceAfter=20
        )

        heading_style = _make_paragraph_style(  # type: ignore[call-arg]
            'CustomHeading',
            'Heading2',
            styles,
            fontSize=self.report_config['fonts']['heading'],
            textColor=self.report_config['colors']['secondary'],
            spaceAfter=12
        )
        
        # Title
        if REPORTLAB_AVAILABLE:
            # Paragraph expects a PropertySet; at runtime title_style is correct when ReportLab present
            story.append(Paragraph("CircuitCheck Analysis Report", cast(Any, title_style)))  # type: ignore[arg-type]
        else:
            # Fallback: use default style from fallback stylesheet
            story.append(Paragraph("CircuitCheck Analysis Report", cast(Any, styles['Title'])))  # type: ignore[arg-type]
        story.append(Spacer(1, 20))
        
        # Summary section
        self._add_summary_section(story, test_result, styles)
        
        # Classification results
        self._add_classification_section(story, test_result, styles, heading_style)
        
        # Component information
        self._add_component_section(story, test_result, styles, heading_style)
        
        # Analysis details
        self._add_analysis_details_section(story, test_result, styles, heading_style)
        
        # Anomalies section
        self._add_anomalies_section(story, test_result, styles, heading_style)
        
        # Images section
        if include_images and test_result.get('image_path'):
            self._add_images_section(story, test_result, styles, heading_style)
        
        # Raw data section
        if include_raw_data:
            self._add_raw_data_section(story, test_result, styles, heading_style)
        
        # Recommendations
        self._add_recommendations_section(story, test_result, styles, heading_style)
        
        # Footer
        self._add_footer_section(story, test_result, styles)
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {filepath}")
        
        return filepath
    
    def _add_summary_section(self, story: List, test_result: Dict, styles):
        """Add summary section to PDF report"""
        
        # Summary table data
        classification = test_result.get('classification', 'UNKNOWN')
        confidence = test_result.get('confidence', 0.0)
        test_date = test_result.get('created_at', datetime.now().isoformat())
        
        # Color code classification
        class_color = self.report_config['colors']['success']
        if classification == 'FAIL':
            class_color = self.report_config['colors']['error']
        elif classification == 'SUSPECT':
            class_color = self.report_config['colors']['warning']
        
        summary_data = [
            ['Test ID', test_result.get('test_id', 'N/A')],
            ['Classification', classification],
            ['Confidence', f"{confidence:.1%}"],
            ['Test Date', test_date[:19].replace('T', ' ')],
            ['User', test_result.get('user_name', 'N/A')]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*get_inch(), 3*get_inch()])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.report_config['colors']['background']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.report_config['fonts']['body']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.report_config['colors']['background']]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Highlight classification row
            ('TEXTCOLOR', (1, 1), (1, 1), class_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
    
    def _add_classification_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add classification details section"""
        
        story.append(Paragraph("Classification Results", heading_style))
        
        # Results explanation
        explanation = test_result.get('analysis_results', {}).get('fusion_analysis', {}).get('explanation', 'No detailed explanation available.')
        story.append(Paragraph(explanation, styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Modality breakdown if available
        analysis_results = test_result.get('analysis_results', {})
        modality_data = []
        
        if 'image_analysis' in analysis_results:
            img_conf = analysis_results['image_analysis'].get('confidence', 0)
            modality_data.append(['Visual Analysis', f"{img_conf:.1%}", self._get_confidence_rating(img_conf)])
        
        if 'electrical_analysis' in analysis_results:
            elec_conf = analysis_results['electrical_analysis'].get('confidence', 0)
            modality_data.append(['Electrical Analysis', f"{elec_conf:.1%}", self._get_confidence_rating(elec_conf)])
        
        if modality_data:
            modality_data.insert(0, ['Analysis Type', 'Confidence', 'Rating'])
            modality_table = Table(modality_data, colWidths=[2*get_inch(), 1.5*get_inch(), 1.5*get_inch()])
            modality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.report_config['colors']['primary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), self.report_config['fonts']['body']),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.report_config['colors']['background']]),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(modality_table)
        
        story.append(Spacer(1, 20))
    
    def _add_component_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add component information section"""
        
        story.append(Paragraph("Component Information", heading_style))
        
        component_data = [
            ['Part Number', test_result.get('part_number', 'N/A')],
            ['Manufacturer', test_result.get('manufacturer', 'N/A')],
            ['Description', test_result.get('description', 'N/A')],
            ['Category', test_result.get('category', 'N/A')],
            ['Package Type', test_result.get('package_type', 'N/A')]
        ]
        
        component_table = Table(component_data, colWidths=[2*get_inch(), 3*get_inch()])
        component_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.report_config['colors']['background']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.report_config['fonts']['body']),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(component_table)
        story.append(Spacer(1, 20))
    
    def _add_analysis_details_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add detailed analysis results"""
        
        story.append(Paragraph("Analysis Details", heading_style))
        
        analysis_results = test_result.get('analysis_results', {})
        
        # Image analysis details
        if 'image_analysis' in analysis_results:
            story.append(Paragraph("<b>Visual Analysis:</b>", styles['Normal']))
            img_analysis = analysis_results['image_analysis']
            story.append(Paragraph(f"Confidence: {img_analysis.get('confidence', 0):.1%}", styles['Normal']))
            
            anomaly_count = len(img_analysis.get('anomalies', []))
            story.append(Paragraph(f"Anomalies detected: {anomaly_count}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Electrical analysis details
        if 'electrical_analysis' in analysis_results:
            story.append(Paragraph("<b>Electrical Analysis:</b>", styles['Normal']))
            elec_analysis = analysis_results['electrical_analysis']
            story.append(Paragraph(f"Confidence: {elec_analysis.get('confidence', 0):.1%}", styles['Normal']))
            
            anomaly_count = len(elec_analysis.get('anomalies', []))
            story.append(Paragraph(f"Anomalies detected: {anomaly_count}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 10))
    
    def _add_anomalies_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add anomalies section to report"""
        
        story.append(Paragraph("Detected Anomalies", heading_style))
        
        # Collect all anomalies
        all_anomalies = []
        analysis_results = test_result.get('analysis_results', {})
        
        # Image anomalies
        img_anomalies = analysis_results.get('image_analysis', {}).get('anomalies', [])
        for anomaly in img_anomalies:
            anomaly['source'] = 'Visual'
            all_anomalies.append(anomaly)
        
        # Electrical anomalies
        elec_anomalies = analysis_results.get('electrical_analysis', {}).get('anomalies', [])
        for anomaly in elec_anomalies:
            anomaly['source'] = 'Electrical'
            all_anomalies.append(anomaly)
        
        if not all_anomalies:
            story.append(Paragraph("No anomalies detected.", styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Create anomalies table
        anomaly_data = [['Source', 'Type', 'Severity', 'Description']]
        
        for anomaly in all_anomalies:
            anomaly_data.append([
                anomaly.get('source', 'Unknown'),
                anomaly.get('type', 'Unknown'),
                anomaly.get('severity', 'Unknown'),
                anomaly.get('description', 'No description')[:60] + ('...' if len(anomaly.get('description', '')) > 60 else '')
            ])
        
        anomaly_table = Table(anomaly_data, colWidths=[1*get_inch(), 1.5*get_inch(), 1*get_inch(), 2.5*get_inch()])
        
        # Style table with color coding for severity
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), self.report_config['colors']['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.report_config['fonts']['body']),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]
        
        # Color code severity
        for i, anomaly in enumerate(all_anomalies, start=1):
            severity = anomaly.get('severity', 'low')
            if severity == 'high':
                table_style.append(('BACKGROUND', (2, i), (2, i), self.report_config['colors']['error']))
                table_style.append(('TEXTCOLOR', (2, i), (2, i), colors.white))
            elif severity == 'medium':
                table_style.append(('BACKGROUND', (2, i), (2, i), self.report_config['colors']['warning']))
        
        anomaly_table.setStyle(TableStyle(table_style))
        story.append(anomaly_table)
        story.append(Spacer(1, 20))
    
    def _add_images_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add component images to report"""
        
        story.append(Paragraph("Component Images", heading_style))
        
        image_path = test_result.get('image_path')
        if not image_path or not os.path.exists(image_path):
            story.append(Paragraph("Component image not available.", styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        try:
            # Add component image
            img = Image(image_path, width=4*get_inch(), height=3*get_inch())
            story.append(img)
            story.append(Paragraph("Component under test", styles['Caption']))
            story.append(Spacer(1, 20))
        except Exception as e:
            story.append(Paragraph(f"Error loading image: {str(e)}", styles['Normal']))
            story.append(Spacer(1, 20))
    
    def _add_raw_data_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add raw measurement data section"""
        
        story.append(PageBreak())
        story.append(Paragraph("Raw Measurement Data", heading_style))
        
        # Get electrical measurements if available
        electrical_data = test_result.get('electrical_measurements', {})
        
        if not electrical_data:
            story.append(Paragraph("No raw measurement data available.", styles['Normal']))
            return
        
        for measurement_type, measurements in electrical_data.items():
            story.append(Paragraph(f"<b>{measurement_type.replace('_', ' ').title()}</b>", styles['Normal']))
            
            if isinstance(measurements, dict):
                measurement_data = [['Pin Combination', 'Value', 'Unit']]
                for pin_combo, value in measurements.items():
                    unit = self._get_unit_for_measurement(measurement_type)
                    measurement_data.append([pin_combo, f"{value:.6g}", unit])
                
                measurement_table = Table(measurement_data, colWidths=[2*get_inch(), 1.5*get_inch(), 1*get_inch()])
                measurement_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.report_config['colors']['background']),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), self.report_config['fonts']['caption']),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(measurement_table)
            
            story.append(Spacer(1, 10))
    
    def _add_recommendations_section(self, story: List, test_result: Dict, styles, heading_style):
        """Add recommendations section"""
        
        story.append(Paragraph("Recommendations", heading_style))
        
        classification = test_result.get('classification', 'UNKNOWN')
        confidence = test_result.get('confidence', 0.0)
        
        recommendations = []
        
        if classification == 'PASS':
            recommendations.append("Component appears genuine and meets expected specifications.")
            recommendations.append("Suitable for use in production applications.")
            if confidence < 0.8:
                recommendations.append("Consider additional testing if higher confidence is required.")
        
        elif classification == 'SUSPECT':
            recommendations.append("Component shows suspicious characteristics that require further investigation.")
            recommendations.append("Recommended actions:")
            recommendations.append("• Perform additional electrical testing")
            recommendations.append("• Compare with known genuine samples")
            recommendations.append("• Verify supply chain documentation")
            recommendations.append("• Consider alternative sourcing")
        
        elif classification == 'FAIL':
            recommendations.append("Component likely counterfeit - DO NOT USE in production.")
            recommendations.append("Immediate actions required:")
            recommendations.append("• Remove from inventory")
            recommendations.append("• Investigate supplier")
            recommendations.append("• Report to authorities if applicable")
            recommendations.append("• Source replacement from trusted supplier")
        
        else:
            recommendations.append("Unable to determine component authenticity.")
            recommendations.append("Recommend manual inspection and additional testing.")
        
        for recommendation in recommendations:
            story.append(Paragraph(recommendation, styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    def _add_footer_section(self, story: List, test_result: Dict, styles):
        """Add footer information"""
        
        if REPORTLAB_AVAILABLE:
            # Resolve parent safely to avoid passing fallback objects
            normal_parent_raw = _safe_get_style(styles, 'Normal')
            normal_parent = _resolve_parent_style(normal_parent_raw)
            footer_style = ParagraphStyle(  # type: ignore[call-arg]
                'Footer',
                parent=cast(Any, normal_parent),  # type: ignore[arg-type]
                fontSize=self.report_config['fonts']['caption'],
                textColor=self.report_config['colors']['secondary'],
                alignment=get_ta_center()
            )
        else:
            # Fallback: reuse existing caption style object
            footer_style = styles['Caption']

        story.append(Spacer(1, 40))
        if REPORTLAB_AVAILABLE:
            story.append(Paragraph("—" * 80, cast(Any, footer_style)))  # type: ignore[arg-type]
            story.append(Paragraph("CircuitCheck Analysis Report", cast(Any, footer_style)))  # type: ignore[arg-type]
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", cast(Any, footer_style)))  # type: ignore[arg-type]
            story.append(Paragraph("This report is computer-generated and may contain technical limitations.", cast(Any, footer_style)))  # type: ignore[arg-type]
        else:
            story.append(Paragraph("—" * 80, cast(Any, footer_style)))  # type: ignore[arg-type]
            story.append(Paragraph("CircuitCheck Analysis Report", cast(Any, footer_style)))  # type: ignore[arg-type]
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", cast(Any, footer_style)))  # type: ignore[arg-type]
            story.append(Paragraph("This report is computer-generated and may contain technical limitations.", cast(Any, footer_style)))  # type: ignore[arg-type]
    
    def _generate_html_report(
        self, 
        test_result: Dict[str, Any], 
        include_images: bool = True,
        include_raw_data: bool = False
    ) -> str:
        """Generate HTML report as fallback"""
        
        test_id = test_result.get('test_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{test_id}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        # Generate HTML content
        html_content = self._generate_html_content(test_result, include_images, include_raw_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")
        return filepath
    
    def _generate_html_content(
        self, 
        test_result: Dict[str, Any], 
        include_images: bool,
        include_raw_data: bool
    ) -> str:
        """Generate HTML content for report"""
        
        classification = test_result.get('classification', 'UNKNOWN')
        confidence = test_result.get('confidence', 0.0)
        
        # Color for classification
        class_color = '#4caf50'  # Green for PASS
        if classification == 'FAIL':
            class_color = '#f44336'  # Red
        elif classification == 'SUSPECT':
            class_color = '#ff9800'  # Orange
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CircuitCheck Analysis Report - {test_result.get('test_id', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #1976d2; color: white; padding: 20px; text-align: center; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .classification {{ font-size: 24px; color: {class_color}; font-weight: bold; }}
                .section {{ margin: 20px 0; }}
                .section h2 {{ color: #424242; border-bottom: 2px solid #1976d2; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #1976d2; color: white; }}
                .anomaly-high {{ background-color: #ffebee; }}
                .anomaly-medium {{ background-color: #fff3e0; }}
                .anomaly-low {{ background-color: #f3e5f5; }}
                .footer {{ text-align: center; margin-top: 40px; padding: 20px; background-color: #f5f5f5; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                .image-container img {{ max-width: 500px; max-height: 400px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CircuitCheck Analysis Report</h1>
                <p>Component Authenticity Analysis</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Test ID:</strong> {test_result.get('test_id', 'N/A')}</p>
                <p><strong>Component:</strong> {test_result.get('part_number', 'N/A')}</p>
                <p><strong>Classification:</strong> <span class="classification">{classification}</span></p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Test Date:</strong> {test_result.get('created_at', 'N/A')[:19]}</p>
            </div>
        """
        
        # Add component details
        html += f"""
            <div class="section">
                <h2>Component Information</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Part Number</td><td>{test_result.get('part_number', 'N/A')}</td></tr>
                    <tr><td>Manufacturer</td><td>{test_result.get('manufacturer', 'N/A')}</td></tr>
                    <tr><td>Description</td><td>{test_result.get('description', 'N/A')}</td></tr>
                    <tr><td>Category</td><td>{test_result.get('category', 'N/A')}</td></tr>
                    <tr><td>Package</td><td>{test_result.get('package_type', 'N/A')}</td></tr>
                </table>
            </div>
        """
        
        # Add analysis results
        analysis_results = test_result.get('analysis_results', {})
        if analysis_results:
            html += '<div class="section"><h2>Analysis Results</h2>'
            
            # Fusion explanation
            explanation = analysis_results.get('fusion_analysis', {}).get('explanation', 'No detailed explanation available.')
            html += f'<p><strong>Analysis Summary:</strong> {explanation}</p>'
            
            # Modality results table
            html += '<table><tr><th>Analysis Type</th><th>Confidence</th><th>Rating</th></tr>'
            
            if 'image_analysis' in analysis_results:
                img_conf = analysis_results['image_analysis'].get('confidence', 0)
                html += f'<tr><td>Visual Analysis</td><td>{img_conf:.1%}</td><td>{self._get_confidence_rating(img_conf)}</td></tr>'
            
            if 'electrical_analysis' in analysis_results:
                elec_conf = analysis_results['electrical_analysis'].get('confidence', 0)
                html += f'<tr><td>Electrical Analysis</td><td>{elec_conf:.1%}</td><td>{self._get_confidence_rating(elec_conf)}</td></tr>'
            
            html += '</table></div>'
        
        # Add anomalies
        html += self._generate_anomalies_html(test_result)
        
        # Add images if requested
        if include_images and test_result.get('image_path'):
            html += f"""
                <div class="section">
                    <h2>Component Images</h2>
                    <div class="image-container">
                        <img src="{test_result.get('image_path')}" alt="Component Image">
                        <p>Component under test</p>
                    </div>
                </div>
            """
        
        # Add raw data if requested
        if include_raw_data:
            html += self._generate_raw_data_html(test_result)
        
        # Add recommendations
        html += self._generate_recommendations_html(test_result)
        
        # Footer
        html += f"""
            <div class="footer">
                <p>CircuitCheck Analysis Report</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>This report is computer-generated and may contain technical limitations.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_json_report(self, test_result: Dict[str, Any], include_raw_data: bool) -> str:
        """Generate JSON report"""
        
        test_id = test_result.get('test_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{test_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare report data
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'report_type': 'component_analysis'
            },
            'test_summary': {
                'test_id': test_result.get('test_id'),
                'classification': test_result.get('classification'),
                'confidence': test_result.get('confidence'),
                'test_date': test_result.get('created_at'),
                'user': test_result.get('user_name')
            },
            'component_info': {
                'part_number': test_result.get('part_number'),
                'manufacturer': test_result.get('manufacturer'),
                'description': test_result.get('description'),
                'category': test_result.get('category'),
                'package_type': test_result.get('package_type')
            },
            'analysis_results': test_result.get('analysis_results', {}),
            'image_path': test_result.get('image_path')
        }
        
        # Include raw data if requested
        if include_raw_data:
            report_data['raw_measurements'] = test_result.get('electrical_measurements', {})
        
        # Write JSON file
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report generated: {filepath}")
        return filepath
    
    def _generate_fallback_report(self, test_result: Dict[str, Any], format: str) -> str:
        """Generate a simple text-based report when ReportLab is not available"""
        
        test_id = test_result.get('test_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{test_id}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("CIRCUITCHECK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("TEST SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test ID: {test_result.get('test_id', 'N/A')}\n")
            f.write(f"Classification: {test_result.get('classification', 'UNKNOWN')}\n")
            f.write(f"Confidence: {test_result.get('confidence', 0):.1%}\n")
            f.write(f"Test Date: {test_result.get('created_at', 'N/A')}\n\n")
            
            f.write("COMPONENT INFORMATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"Part Number: {test_result.get('part_number', 'N/A')}\n")
            f.write(f"Manufacturer: {test_result.get('manufacturer', 'N/A')}\n")
            f.write(f"Description: {test_result.get('description', 'N/A')}\n")
            f.write(f"Category: {test_result.get('category', 'N/A')}\n\n")
            
            # Add analysis explanation if available
            analysis_results = test_result.get('analysis_results', {})
            explanation = analysis_results.get('fusion_analysis', {}).get('explanation', 'No detailed explanation available.')
            f.write("ANALYSIS EXPLANATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"{explanation}\n\n")
            
            f.write("REPORT GENERATED\n")
            f.write("-" * 20 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Note: This is a fallback text report. Install ReportLab for full PDF reports.\n")
        
        logger.info(f"Fallback text report generated: {filepath}")
        return filepath
    
    def generate_batch_report(
        self, 
        test_results: List[Dict[str, Any]], 
        format: str = 'pdf'
    ) -> str:
        """Generate a batch report for multiple test results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_report_{timestamp}.{format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if format.lower() == 'csv':
            return self._generate_csv_batch_report(test_results, filepath)
        elif format.lower() == 'json':
            return self._generate_json_batch_report(test_results, filepath)
        else:
            raise ValueError(f"Batch reports only support CSV and JSON formats, got: {format}")
    
    def _generate_csv_batch_report(self, test_results: List[Dict[str, Any]], filepath: str) -> str:
        """Generate CSV batch report"""
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'test_id', 'part_number', 'manufacturer', 'classification', 
                'confidence', 'test_date', 'user_name', 'image_analysis_confidence',
                'electrical_analysis_confidence', 'anomaly_count'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in test_results:
                analysis_results = result.get('analysis_results', {})
                
                # Count total anomalies
                img_anomalies = len(analysis_results.get('image_analysis', {}).get('anomalies', []))
                elec_anomalies = len(analysis_results.get('electrical_analysis', {}).get('anomalies', []))
                total_anomalies = img_anomalies + elec_anomalies
                
                row = {
                    'test_id': result.get('test_id', ''),
                    'part_number': result.get('part_number', ''),
                    'manufacturer': result.get('manufacturer', ''),
                    'classification': result.get('classification', ''),
                    'confidence': result.get('confidence', 0),
                    'test_date': result.get('created_at', ''),
                    'user_name': result.get('user_name', ''),
                    'image_analysis_confidence': analysis_results.get('image_analysis', {}).get('confidence', ''),
                    'electrical_analysis_confidence': analysis_results.get('electrical_analysis', {}).get('confidence', ''),
                    'anomaly_count': total_anomalies
                }
                
                writer.writerow(row)
        
        logger.info(f"CSV batch report generated: {filepath}")
        return filepath
    
    def _generate_json_batch_report(self, test_results: List[Dict[str, Any]], filepath: str) -> str:
        """Generate JSON batch report"""
        
        batch_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'batch_analysis',
                'total_tests': len(test_results)
            },
            'summary_statistics': self._calculate_batch_statistics(test_results),
            'test_results': test_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        
        logger.info(f"JSON batch report generated: {filepath}")
        return filepath
    
    def _calculate_batch_statistics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for batch report"""
        
        if not test_results:
            return {}
        
        classifications = [r.get('classification', 'UNKNOWN') for r in test_results]
        confidences = [r.get('confidence', 0) for r in test_results]
        
        stats = {
            'total_tests': len(test_results),
            'classification_counts': {
                'PASS': classifications.count('PASS'),
                'SUSPECT': classifications.count('SUSPECT'),
                'FAIL': classifications.count('FAIL'),
                'UNKNOWN': classifications.count('UNKNOWN')
            },
            'pass_rate': classifications.count('PASS') / len(test_results),
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'date_range': {
                'earliest': min([r.get('created_at', '') for r in test_results if r.get('created_at')], default=''),
                'latest': max([r.get('created_at', '') for r in test_results if r.get('created_at')], default='')
            }
        }
        
        return stats
    
    # Utility methods
    def _get_confidence_rating(self, confidence: float) -> str:
        """Convert confidence score to rating"""
        if confidence >= 0.9:
            return "Excellent"
        elif confidence >= 0.8:
            return "Good"
        elif confidence >= 0.7:
            return "Fair"
        elif confidence >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_unit_for_measurement(self, measurement_type: str) -> str:
        """Get unit for measurement type"""
        unit_map = {
            'resistance': 'Ω',
            'capacitance': 'pF',
            'leakage_current': 'μA',
            'timing': 'ns'
        }
        return unit_map.get(measurement_type, '')
    
    def _generate_anomalies_html(self, test_result: Dict[str, Any]) -> str:
        """Generate HTML for anomalies section"""
        
        html = '<div class="section"><h2>Detected Anomalies</h2>'
        
        # Collect all anomalies
        all_anomalies = []
        analysis_results = test_result.get('analysis_results', {})
        
        # Image anomalies
        img_anomalies = analysis_results.get('image_analysis', {}).get('anomalies', [])
        for anomaly in img_anomalies:
            anomaly['source'] = 'Visual'
            all_anomalies.append(anomaly)
        
        # Electrical anomalies
        elec_anomalies = analysis_results.get('electrical_analysis', {}).get('anomalies', [])
        for anomaly in elec_anomalies:
            anomaly['source'] = 'Electrical'
            all_anomalies.append(anomaly)
        
        if not all_anomalies:
            html += '<p>No anomalies detected.</p></div>'
            return html
        
        html += '<table><tr><th>Source</th><th>Type</th><th>Severity</th><th>Description</th></tr>'
        
        for anomaly in all_anomalies:
            severity = anomaly.get('severity', 'low')
            css_class = f'anomaly-{severity}'
            
            html += f'''<tr class="{css_class}">
                <td>{anomaly.get('source', 'Unknown')}</td>
                <td>{anomaly.get('type', 'Unknown')}</td>
                <td>{severity.title()}</td>
                <td>{anomaly.get('description', 'No description')}</td>
            </tr>'''
        
        html += '</table></div>'
        return html
    
    def _generate_raw_data_html(self, test_result: Dict[str, Any]) -> str:
        """Generate HTML for raw data section"""
        
        html = '<div class="section"><h2>Raw Measurement Data</h2>'
        
        electrical_data = test_result.get('electrical_measurements', {})
        
        if not electrical_data:
            html += '<p>No raw measurement data available.</p></div>'
            return html
        
        for measurement_type, measurements in electrical_data.items():
            html += f'<h3>{measurement_type.replace("_", " ").title()}</h3>'
            html += '<table><tr><th>Pin Combination</th><th>Value</th><th>Unit</th></tr>'
            
            if isinstance(measurements, dict):
                for pin_combo, value in measurements.items():
                    unit = self._get_unit_for_measurement(measurement_type)
                    html += f'<tr><td>{pin_combo}</td><td>{value:.6g}</td><td>{unit}</td></tr>'
            
            html += '</table>'
        
        html += '</div>'
        return html
    
    def _generate_recommendations_html(self, test_result: Dict[str, Any]) -> str:
        """Generate HTML for recommendations section"""
        
        html = '<div class="section"><h2>Recommendations</h2>'
        
        classification = test_result.get('classification', 'UNKNOWN')
        confidence = test_result.get('confidence', 0.0)
        
        if classification == 'PASS':
            html += '<p>Component appears genuine and meets expected specifications.</p>'
            html += '<p>Suitable for use in production applications.</p>'
            if confidence < 0.8:
                html += '<p><em>Consider additional testing if higher confidence is required.</em></p>'
        
        elif classification == 'SUSPECT':
            html += '<p><strong>Component shows suspicious characteristics that require further investigation.</strong></p>'
            html += '<p>Recommended actions:</p>'
            html += '<ul>'
            html += '<li>Perform additional electrical testing</li>'
            html += '<li>Compare with known genuine samples</li>'
            html += '<li>Verify supply chain documentation</li>'
            html += '<li>Consider alternative sourcing</li>'
            html += '</ul>'
        
        elif classification == 'FAIL':
            html += '<p style="color: red;"><strong>Component likely counterfeit - DO NOT USE in production.</strong></p>'
            html += '<p>Immediate actions required:</p>'
            html += '<ul>'
            html += '<li>Remove from inventory</li>'
            html += '<li>Investigate supplier</li>'
            html += '<li>Report to authorities if applicable</li>'
            html += '<li>Source replacement from trusted supplier</li>'
            html += '</ul>'
        
        else:
            html += '<p>Unable to determine component authenticity.</p>'
            html += '<p>Recommend manual inspection and additional testing.</p>'
        
        html += '</div>'
        return html


def main():
    """Example usage of the report generator"""
    
    # Create sample test result
    sample_test_result = {
        'test_id': 'TEST_2024_001',
        'part_number': 'MC74HC00AN',
        'manufacturer': 'ON Semiconductor',
        'description': 'Quad 2-input NAND gate',
        'category': 'Logic IC',
        'package_type': 'DIP-14',
        'classification': 'SUSPECT',
        'confidence': 0.72,
        'created_at': '2024-01-15T10:30:00Z',
        'user_name': 'test_engineer',
        'image_path': '/uploads/demo/mc74hc00an_suspect.jpg',
        'analysis_results': {
            'image_analysis': {
                'confidence': 0.68,
                'anomalies': [
                    {
                        'type': 'marking_inconsistency',
                        'severity': 'medium',
                        'description': 'Font style differs from expected specification'
                    }
                ]
            },
            'electrical_analysis': {
                'confidence': 0.76,
                'anomalies': [
                    {
                        'type': 'resistance_deviation',
                        'severity': 'low',
                        'description': 'Pin resistance slightly outside normal range'
                    }
                ]
            },
            'fusion_analysis': {
                'explanation': 'Component shows minor anomalies in both visual and electrical analysis that warrant closer inspection'
            }
        },
        'electrical_measurements': {
            'resistance': {
                'pin1_pin14': 1050000,
                'pin7_pin14': 0.52
            },
            'capacitance': {
                'pin1_gnd': 5.3,
                'pin14_gnd': 10.2
            },
            'leakage_current': {
                'pin1': 0.0012,
                'pin14': 0.0015
            },
            'timing': {
                'rise_time': 6.8,
                'fall_time': 6.5,
                'propagation_delay': 9.5
            }
        }
    }
    
    # Initialize report generator
    generator = ReportGenerator()
    
    print("Generating sample reports...")
    
    # Generate PDF report
    try:
        pdf_path = generator.generate_analysis_report(
            sample_test_result, 
            format='pdf',
            include_images=False,  # Set to False for demo
            include_raw_data=True
        )
        print(f"PDF report generated: {pdf_path}")
    except Exception as e:
        print(f"PDF generation failed: {e}")
    
    # Generate HTML report
    try:
        html_path = generator.generate_analysis_report(
            sample_test_result,
            format='html',
            include_images=False,
            include_raw_data=True
        )
        print(f"HTML report generated: {html_path}")
    except Exception as e:
        print(f"HTML generation failed: {e}")
    
    # Generate JSON report
    try:
        json_path = generator.generate_analysis_report(
            sample_test_result,
            format='json',
            include_raw_data=True
        )
        print(f"JSON report generated: {json_path}")
    except Exception as e:
        print(f"JSON generation failed: {e}")
    
    # Generate batch report
    try:
        batch_csv_path = generator.generate_batch_report([sample_test_result], format='csv')
        print(f"Batch CSV report generated: {batch_csv_path}")
    except Exception as e:
        print(f"Batch CSV generation failed: {e}")
    
    print("\nReport generation demo completed!")


if __name__ == "__main__":
    main()