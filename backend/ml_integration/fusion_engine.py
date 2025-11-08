import numpy as np
import logging

logger = logging.getLogger(__name__)

def combine_results(image_result, electrical_result):
    """
    Combine results from image and electrical analyses into a final result
    
    Args:
        image_result: Dict with image analysis results
        electrical_result: Dict with electrical analysis results
        
    Returns:
        Dict with combined results including:
        - confidence: Overall confidence score (0-1)
        - classification: PASS/SUSPECT/FAIL
        - anomalies: Combined list of anomalies
    """
    try:
        # Extract confidence scores
        image_confidence = image_result.get('confidence', 0)
        electrical_confidence = electrical_result.get('confidence', 0)
        
        # Extract classifications
        image_classification = image_result.get('classification', 'FAIL')
        electrical_classification = electrical_result.get('classification', 'FAIL')
        
        # Combine confidence scores with weights
        # Adjust weights based on which modality is more reliable for your use case
        image_weight = 0.6  # Image analysis has 60% weight
        electrical_weight = 0.4  # Electrical analysis has 40% weight
        
        combined_confidence = (image_confidence * image_weight + 
                              electrical_confidence * electrical_weight)
        
        # Determine final classification
        # Use the more conservative classification (FAIL > SUSPECT > PASS)
        if 'FAIL' in [image_classification, electrical_classification]:
            final_classification = 'FAIL'
        elif 'SUSPECT' in [image_classification, electrical_classification]:
            final_classification = 'SUSPECT'
        else:
            final_classification = 'PASS'
        
        # Combine anomalies
        anomalies = {
            'visual': image_result.get('anomalies', []),
            'electrical': electrical_result.get('outliers', [])
        }
        
        # Create explanation
        explanation = _generate_explanation(
            image_classification, electrical_classification,
            image_confidence, electrical_confidence,
            anomalies
        )
        
        logger.info(f"Fusion results - Final classification: {final_classification} " 
                   f"(Score: {combined_confidence:.4f})")
        
        return {
            'confidence': float(combined_confidence),
            'classification': final_classification,
            'anomalies': anomalies,
            'explanation': explanation,
            'image_result': {
                'confidence': image_confidence,
                'classification': image_classification
            },
            'electrical_result': {
                'confidence': electrical_confidence,
                'classification': electrical_classification
            }
        }
    
    except Exception as e:
        logger.error(f"Error in fusion engine: {e}")
        return {
            'confidence': 0,
            'classification': 'FAIL',
            'error': str(e),
            'explanation': f"Error combining results: {str(e)}"
        }

def _generate_explanation(image_class, electrical_class, image_conf, electrical_conf, anomalies):
    """Generate a human-readable explanation of the results"""
    
    explanation = []
    
    # Overall assessment
    if image_class == 'FAIL' and electrical_class == 'FAIL':
        explanation.append("Component failed both visual and electrical tests.")
    elif image_class == 'PASS' and electrical_class == 'PASS':
        explanation.append("Component passed both visual and electrical tests.")
    elif image_class == 'FAIL':
        explanation.append("Component failed visual inspection but passed electrical tests.")
    elif electrical_class == 'FAIL':
        explanation.append("Component failed electrical tests but passed visual inspection.")
    else:
        explanation.append("Component showed suspect indicators in testing.")
    
    # Visual details
    visual_anomalies = anomalies.get('visual', [])
    if visual_anomalies:
        explanation.append(f"Found {len(visual_anomalies)} visual anomalies:")
        anomaly_types = {}
        for anomaly in visual_anomalies[:5]:  # Limit to 5 examples
            anomaly_type = anomaly.get('type', 'unknown')
            anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
        
        for anomaly_type, count in anomaly_types.items():
            explanation.append(f"- {count} instances of {anomaly_type.replace('_', ' ')}")
    else:
        explanation.append("No visual anomalies detected.")
    
    # Electrical details
    electrical_anomalies = anomalies.get('electrical', [])
    if electrical_anomalies:
        explanation.append(f"Found {len(electrical_anomalies)} electrical anomalies:")
        for anomaly in electrical_anomalies[:5]:  # Limit to 5 examples
            feature = anomaly.get('feature', 'unknown')
            expected = anomaly.get('expected', 0)
            actual = anomaly.get('actual', 0)
            deviation = anomaly.get('deviation', 0) * 100  # Convert to percentage
            explanation.append(f"- {feature}: expected {expected:.2f}, got {actual:.2f} " 
                             f"({deviation:.1f}% deviation)")
    else:
        explanation.append("No electrical anomalies detected.")
    
    # Confidence assessment
    explanation.append(f"Visual confidence: {image_conf:.2f}, Electrical confidence: {electrical_conf:.2f}")
    
    return explanation