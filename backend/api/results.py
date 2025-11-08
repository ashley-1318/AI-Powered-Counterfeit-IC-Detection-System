from flask import Blueprint, request, jsonify, current_app, send_file
import os
import logging
import json
from datetime import datetime
import io

# Optional imports for PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter as rl_letter, A4 as rl_A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
    
    # Helper functions for ReportLab constants
    def get_letter():
        return rl_letter
    
    def get_A4():
        return rl_A4
        
    def get_rl_colors():
        return rl_colors
        
except ImportError:
    REPORTLAB_AVAILABLE = False
    
    # Create mock classes for when ReportLab is not available
    class MockCanvas:
        def __init__(self, *args, **kwargs):
            pass
        def setFont(self, *args, **kwargs):
            pass
        def drawString(self, *args, **kwargs):
            pass
        def setFillColor(self, *args, **kwargs):
            pass
        def showPage(self):
            pass
        def save(self):
            pass
    
    class MockCanvasModule:
        Canvas = MockCanvas
    
    canvas = MockCanvasModule()
    
    # Fallback implementations
    def get_letter_fallback():
        return (612, 792)
    
    def get_A4():
        return (595.27, 841.89)
        
    class MockColors:
        black = "#000000"
        blue = "#0000FF"
        red = "#FF0000"
        grey = "#808080"
        green = "#008000"
        orange = "#FFA500"
    
    def get_mock_colors():
        return MockColors()
    
    # Create mock classes for when ReportLab is not available
    class MockSimpleDocTemplate:
        def __init__(self, *args, **kwargs):
            pass
        def build(self, *args, **kwargs):
            pass
    
    SimpleDocTemplate = MockSimpleDocTemplate

from database.db import db
from database.models import TestResult

results_bp = Blueprint('results', __name__)
logger = logging.getLogger(__name__)

@results_bp.route('/test/<int:test_id>', methods=['GET'])
def get_test_result(test_id):
    """Get a specific test result by ID"""
    try:
        test_result = TestResult.query.get(test_id)
        
        if not test_result:
            return jsonify({'error': 'Test result not found'}), 404
            
        # Convert test result to dictionary
        result = {
            'id': test_result.id,
            'component_id': test_result.component_id,
            'user_id': test_result.user_id,
            'test_date': test_result.test_date.isoformat(),
            'image_score': test_result.image_score,
            'electrical_score': test_result.electrical_score,
            'fusion_score': test_result.fusion_score,
            'result_class': test_result.result_class,
            'macro_image_url': f"/api/results/image/{test_result.id}/macro",
            'micro_image_url': f"/api/results/image/{test_result.id}/micro" if test_result.micro_image_path else None,
            'electrical_measurements': test_result.electrical_measurements,
            'anomaly_data': test_result.anomaly_data,
            'notes': test_result.notes,
            'batch_id': test_result.batch_id
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error retrieving test result: {str(e)}")
        return jsonify({'error': str(e)}), 500

@results_bp.route('/recent', methods=['GET'])
def get_recent_results():
    """Get recent test results"""
    try:
        limit = request.args.get('limit', 10, type=int)
        user_id = request.args.get('user_id', type=int)
        component_id = request.args.get('component_id', type=int)
        
        query = TestResult.query
        
        if user_id:
            query = query.filter(TestResult.user_id == user_id)
            
        if component_id:
            query = query.filter(TestResult.component_id == component_id)
            
        results = query.order_by(TestResult.test_date.desc()).limit(limit).all()
        
        # Convert test results to list of dictionaries
        result_list = []
        for test_result in results:
            result = {
                'id': test_result.id,
                'component_id': test_result.component_id,
                'user_id': test_result.user_id,
                'test_date': test_result.test_date.isoformat(),
                'result_class': test_result.result_class,
                'fusion_score': test_result.fusion_score
            }
            result_list.append(result)
        
        return jsonify(result_list)
        
    except Exception as e:
        logger.error(f"Error retrieving recent results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@results_bp.route('/image/<int:test_id>/<image_type>', methods=['GET'])
def get_test_image(test_id, image_type):
    """Get the test image"""
    try:
        test_result = TestResult.query.get(test_id)
        
        if not test_result:
            return jsonify({'error': 'Test result not found'}), 404
            
        if image_type == 'macro':
            image_path = test_result.macro_image_path
        elif image_type == 'micro':
            image_path = test_result.micro_image_path
        else:
            return jsonify({'error': 'Invalid image type'}), 400
            
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
            
        return send_file(image_path)
        
    except Exception as e:
        logger.error(f"Error retrieving test image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@results_bp.route('/report/<int:test_id>', methods=['GET'])
def generate_report(test_id):
    """Generate a PDF report for a test result"""
    if not REPORTLAB_AVAILABLE:
        return jsonify({'error': 'PDF generation not available - ReportLab not installed'}), 500
        
    try:
        test_result = TestResult.query.get(test_id)
        
        if not test_result:
            return jsonify({'error': 'Test result not found'}), 404
            
        if not REPORTLAB_AVAILABLE:
            # Fallback to JSON report when reportlab is not available
            return jsonify({
                'test_id': test_result.id,
                'date': test_result.test_date.isoformat() if test_result.test_date else None,
                'classification': test_result.result_class,
                'image_score': test_result.image_score,
                'electrical_score': test_result.electrical_score,
                'fusion_score': test_result.fusion_score,
                'electrical_measurements': test_result.electrical_measurements,
                'anomaly_data': test_result.anomaly_data,
                'notes': test_result.notes
            })
            
        # Create a PDF in memory
        buffer = io.BytesIO()
        letter_size = get_letter() if REPORTLAB_AVAILABLE else get_letter_fallback()
        colors_obj = get_rl_colors() if REPORTLAB_AVAILABLE else get_mock_colors()
        p = canvas.Canvas(buffer, pagesize=letter_size)
        width, height = letter_size
        
        # Add report title
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, height - 50, "CircuitCheck Component Analysis Report")
        
        # Add test info
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 100, "Test Information")
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 120, f"Test ID: {test_result.id}")
        p.drawString(50, height - 140, f"Date: {test_result.test_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add result classification
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 180, "Analysis Result")
        p.setFont("Helvetica-Bold", 24)
        
        result_color = colors_obj.green if test_result.result_class == 'PASS' else \
                      colors_obj.orange if test_result.result_class == 'SUSPECT' else \
                      colors_obj.red
        p.setFillColor(result_color)
        p.drawString(50, height - 210, test_result.result_class)
        p.setFillColor(colors_obj.black)
        
        # Add confidence scores
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 250, "Confidence Scores")
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 270, f"Overall: {test_result.fusion_score:.2f}")
        p.drawString(50, height - 290, f"Image Analysis: {test_result.image_score:.2f}")
        p.drawString(50, height - 310, f"Electrical Analysis: {test_result.electrical_score:.2f}")
        
        # Add anomaly information
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 350, "Detected Anomalies")
        
        if test_result.anomaly_data:
            y_pos = height - 370
            p.setFont("Helvetica", 12)
            
            # Visual anomalies
            visual_anomalies = test_result.anomaly_data.get('visual', [])
            if visual_anomalies:
                p.drawString(50, y_pos, f"Visual Anomalies: {len(visual_anomalies)}")
                y_pos -= 20
                
                # Count anomaly types
                anomaly_types = {}
                for anomaly in visual_anomalies:
                    anomaly_type = anomaly.get('type', 'unknown')
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
                
                for anomaly_type, count in list(anomaly_types.items())[:3]:  # Show top 3
                    p.drawString(70, y_pos, f"- {count} instances of {anomaly_type.replace('_', ' ')}")
                    y_pos -= 20
                    
                if len(anomaly_types) > 3:
                    p.drawString(70, y_pos, f"- Plus {len(anomaly_types) - 3} more anomaly types")
                    y_pos -= 20
            else:
                p.drawString(50, y_pos, "Visual Anomalies: None detected")
                y_pos -= 20
                
            # Electrical anomalies
            y_pos -= 10
            electrical_anomalies = test_result.anomaly_data.get('electrical', [])
            if electrical_anomalies:
                p.drawString(50, y_pos, f"Electrical Anomalies: {len(electrical_anomalies)}")
                y_pos -= 20
                
                for anomaly in electrical_anomalies[:3]:  # Show top 3
                    feature = anomaly.get('feature', 'unknown')
                    deviation = anomaly.get('deviation', 0) * 100  # Convert to percentage
                    p.drawString(70, y_pos, f"- {feature}: {deviation:.1f}% deviation")
                    y_pos -= 20
                    
                if len(electrical_anomalies) > 3:
                    p.drawString(70, y_pos, f"- Plus {len(electrical_anomalies) - 3} more anomalies")
                    y_pos -= 20
            else:
                p.drawString(50, y_pos, "Electrical Anomalies: None detected")
        else:
            p.setFont("Helvetica", 12)
            p.drawString(50, height - 370, "No anomalies detected")
            
        # Add footer
        p.setFont("Helvetica-Italic", 10)
        p.drawString(50, 40, "Generated by CircuitCheck - AI-Powered Counterfeit Component Detection")
        p.drawString(50, 25, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Save the PDF
        p.showPage()
        p.save()
        buffer.seek(0)
        
        # Return the PDF
        return send_file(
            buffer, 
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'CircuitCheck_Report_{test_id}.pdf'
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500