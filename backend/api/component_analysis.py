from flask import Blueprint, request, jsonify, current_app, abort
import os
from werkzeug.utils import secure_filename
import uuid
import logging
import json
from datetime import datetime

from ml_integration.image_model import analyze_image
# Removed electrical_model import
from ml_integration.fusion_engine import combine_results
from database.db import db
from database.models import Component, TestResult

component_analysis_bp = Blueprint('component_analysis', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@component_analysis_bp.route('/analyze', methods=['POST'])
def analyze_component():
    """
    Endpoint for analyzing component authenticity using visual inspection
    
    Expects:
    - image: file - Component image
    - component_id: int - Component ID for reference data (optional)
    - part_number: string - Component part number
    
    Returns:
    - analysis_result: JSON with classification and confidence scores
    """
    try:
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected image file'}), 400
            
        if not file.filename or not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Generate unique filename and save the image
        filename = secure_filename(file.filename or 'unknown.jpg')
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Get component reference data if component_id provided
        component_ref_data = None
        part_number = request.form.get('part_number', 'Unknown')
        
        if 'component_id' in request.form:
            component = Component.query.get(request.form['component_id'])
            if component:
                component_ref_data = component.reference_data
                part_number = component.part_number
        
        # Run analysis on the image
        image_result = analyze_image(file_path)
        
        # Set default results for electrical analysis (removed)
        electrical_result = {
            'confidence': 1.0,  # Default high confidence since we don't use electrical measures
            'classification': 'PASS',  # Default to pass
            'outliers': []  # No outliers
        }
        
        # Use only image results (no electrical analysis)
        # Modify fusion engine call to use only image data
        final_result = {
            'classification': image_result.get('classification', 'FAIL'),
            'confidence': image_result.get('confidence', 0),
            'anomalies': image_result.get('anomalies', {}),
            'details': {
                'visual_issues': image_result.get('anomalies', {}),
                'image_score': image_result.get('confidence', 0)
            }
        }

        # SIMPLIFIED APPROACH: Always create a new component for every analysis
        # This ensures we never have NULL component_id issues
        
        # First create a component
        new_component = Component()
        new_component.part_number = request.form.get('part_number', 'Unknown_' + str(uuid.uuid4())[:8])
        new_component.manufacturer = request.form.get('manufacturer', 'Unknown')
        new_component.description = "Component created during analysis"
        new_component.category = request.form.get('category', 'Unknown')
        new_component.package_type = request.form.get('package_type', 'Unknown')
        new_component.reference_data = {}
        
        # Add and commit component first to get ID
        db.session.add(new_component)
        db.session.commit()
        
        if not new_component.id:
            logger.error("Failed to create component - aborting")
            abort(500, description="Database error: Failed to create component")
        
        logger.info(f"Created new component with ID: {new_component.id}")
        
        # Now create test result with the new component ID
        test_result = TestResult()
        test_result.component_id = new_component.id  # Use the new component's ID
        
        test_result.user_id = int(request.form.get('user_id', 0)) if request.form.get('user_id') else None
        test_result.test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_result.test_date = datetime.now()
        test_result.image_path = file_path
        test_result.classification = final_result['classification']
        test_result.confidence = final_result['confidence']
        test_result.image_score = image_result.get('confidence', 0)
        test_result.electrical_score = electrical_result.get('confidence', 0)
        test_result.fusion_score = final_result['confidence']
        test_result.result_class = final_result['classification']
        test_result.anomaly_data = json.dumps(final_result.get('anomalies', {}))
        test_result.electrical_measurements = json.dumps({})
        test_result.analysis_results = {
            'image': image_result,
            'result': final_result
        }
        # Log the values before saving to help debug
        logger.info(f"About to save test result with component_id: {test_result.component_id}, " 
                   f"user_id: {test_result.user_id}, test_id: {test_result.test_id}")
        
        # Verify component_id is not None before committing
        if test_result.component_id is None:
            logger.error("component_id is None, creating emergency component")
            emergency_component = Component()
            emergency_component.part_number = "EMERGENCY_" + datetime.now().strftime('%Y%m%d_%H%M%S')
            emergency_component.description = "Emergency component created to prevent NULL constraint"
            emergency_component.category = "Emergency"
            db.session.add(emergency_component)
            db.session.commit()
            test_result.component_id = emergency_component.id
        
        try:
            db.session.add(test_result)
            db.session.commit()
            
            # Add test_id to response
            final_result['test_id'] = test_result.id
            
            logger.info(f"Component analysis completed - Result: {final_result['classification']} " 
                      f"(Score: {final_result['confidence']}), Test Result ID: {test_result.id}")
            
            return jsonify(final_result)
        except Exception as db_error:
            db.session.rollback()
            logger.error(f"Database error during commit: {str(db_error)}")
            raise db_error
    
    except Exception as e:
        logger.error(f"Error during component analysis: {str(e)}")
        # Log more details about what might be happening
        if 'component_id' in str(e) and 'NULL' in str(e):
            logger.error("This appears to be a component_id NULL constraint error.")
            # Log the form data that might contain part number
            logger.error(f"Request form data: {request.form}")
            logger.error(f"Request part_number: {request.form.get('part_number', 'Not provided')}")
            
        return jsonify({'error': str(e)}), 500