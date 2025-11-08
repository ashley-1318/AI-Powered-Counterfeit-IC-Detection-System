"""
Fusion Engine for CircuitCheck
Combines image analysis and electrical signature analysis results
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalFusionEngine:
    """
    Combines results from multiple analysis modalities (image + electrical)
    to provide a comprehensive assessment of component authenticity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.fusion_weights = self.config['fusion_weights']
        self.confidence_thresholds = self.config['confidence_thresholds']
        self.anomaly_weights = self.config['anomaly_weights']
        
        # History for adaptive weighting
        self.analysis_history = []
        self.adaptive_enabled = self.config.get('adaptive_weighting', False)
        
    def _default_config(self) -> Dict:
        """Default configuration for fusion engine"""
        return {
            'fusion_weights': {
                'image': 0.6,      # Visual analysis weight
                'electrical': 0.4,  # Electrical analysis weight
                'context': 0.1      # Context bonus (part number match, etc.)
            },
            'confidence_thresholds': {
                'pass': 0.7,        # Minimum confidence for PASS
                'suspect': 0.4      # Minimum confidence for SUSPECT (below = FAIL)
            },
            'anomaly_weights': {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.2
            },
            'adaptive_weighting': True,
            'disagreement_threshold': 0.3,  # Max difference between modalities
            'min_samples_adaptive': 50      # Minimum samples before adaptive weighting
        }
    
    def fuse_analysis_results(
        self,
        image_result: Optional[Dict] = None,
        electrical_result: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Combine results from different analysis modalities
        
        Args:
            image_result: Result from image analysis
            electrical_result: Result from electrical signature analysis
            context: Additional context (part number, manufacturer, etc.)
            
        Returns:
            Fused analysis result
        """
        
        if not image_result and not electrical_result:
            raise ValueError("At least one analysis result must be provided")
        
        # Initialize fusion result
        fusion_result = {
            'classification': 'UNKNOWN',
            'confidence': 0.0,
            'fusion_weights': self.fusion_weights.copy(),
            'modality_results': {},
            'anomalies': [],
            'explanation': '',
            'timestamp': datetime.now().isoformat(),
            'analysis_id': f"fusion_{int(datetime.now().timestamp())}"
        }
        
        # Store individual modality results
        if image_result:
            fusion_result['modality_results']['image'] = image_result
        if electrical_result:
            fusion_result['modality_results']['electrical'] = electrical_result
        if context:
            fusion_result['modality_results']['context'] = context
        
        # Determine available modalities
        available_modalities = []
        if image_result:
            available_modalities.append('image')
        if electrical_result:
            available_modalities.append('electrical')
        
        # Adjust weights for available modalities
        adjusted_weights = self._adjust_weights(available_modalities, context)
        fusion_result['fusion_weights'] = adjusted_weights
        
        # Calculate fused confidence
        fused_confidence = self._calculate_fused_confidence(
            image_result, electrical_result, adjusted_weights
        )
        
        # Check for modality disagreement
        disagreement_info = self._check_modality_disagreement(
            image_result, electrical_result
        )
        
        # Combine anomalies
        combined_anomalies = self._combine_anomalies(
            image_result, electrical_result
        )
        
        # Apply context adjustments
        context_adjustment = self._calculate_context_adjustment(context)
        final_confidence = min(1.0, max(0.0, fused_confidence + context_adjustment))
        
        # Determine final classification
        final_classification = self._determine_classification(
            final_confidence, disagreement_info, combined_anomalies
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            final_classification, final_confidence, available_modalities,
            disagreement_info, combined_anomalies, context_adjustment
        )
        
        # Populate result
        fusion_result.update({
            'classification': final_classification,
            'confidence': final_confidence,
            'raw_confidence': fused_confidence,
            'context_adjustment': context_adjustment,
            'anomalies': combined_anomalies,
            'disagreement': disagreement_info,
            'explanation': explanation
        })
        
        # Store for adaptive learning
        if self.adaptive_enabled:
            self._update_analysis_history(fusion_result)
        
        return fusion_result
    
    def _adjust_weights(
        self,
        available_modalities: List[str],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Adjust fusion weights based on available modalities and context"""
        
        weights = self.fusion_weights.copy()
        
        # If only one modality is available, give it full weight
        if len(available_modalities) == 1:
            modality = available_modalities[0]
            weights = {k: 0.0 for k in weights}
            weights[modality] = 1.0
            return weights
        
        # Adaptive weighting based on historical performance
        if self.adaptive_enabled and len(self.analysis_history) >= self.config['min_samples_adaptive']:
            weights = self._calculate_adaptive_weights(available_modalities)
        
        # Normalize weights for available modalities
        available_weight_sum = sum(weights[mod] for mod in available_modalities)
        if available_weight_sum > 0:
            for modality in available_modalities:
                weights[modality] = weights[modality] / available_weight_sum
        
        # Zero out unavailable modalities
        for modality in weights:
            if modality not in available_modalities and modality != 'context':
                weights[modality] = 0.0
        
        return weights
    
    def _calculate_fused_confidence(
        self,
        image_result: Optional[Dict],
        electrical_result: Optional[Dict],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted confidence from available modalities"""
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        if image_result and weights['image'] > 0:
            img_confidence = image_result.get('confidence', 0.0)
            weighted_confidence += img_confidence * weights['image']
            total_weight += weights['image']
        
        if electrical_result and weights['electrical'] > 0:
            elec_confidence = electrical_result.get('confidence', 0.0)
            weighted_confidence += elec_confidence * weights['electrical']
            total_weight += weights['electrical']
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return 0.5  # Neutral confidence if no valid results
    
    def _check_modality_disagreement(
        self,
        image_result: Optional[Dict],
        electrical_result: Optional[Dict]
    ) -> Dict[str, Any]:
        """Check for significant disagreement between modalities"""
        
        if not image_result or not electrical_result:
            return {'has_disagreement': False}
        
        img_conf = image_result.get('confidence', 0.5)
        elec_conf = electrical_result.get('confidence', 0.5)
        img_class = image_result.get('classification', 'UNKNOWN')
        elec_class = electrical_result.get('classification', 'UNKNOWN')
        
        # Check classification disagreement
        class_disagreement = img_class != elec_class and img_class != 'UNKNOWN' and elec_class != 'UNKNOWN'
        
        # Check confidence disagreement
        conf_difference = abs(img_conf - elec_conf)
        conf_disagreement = conf_difference > self.config['disagreement_threshold']
        
        has_disagreement = class_disagreement or conf_disagreement
        
        return {
            'has_disagreement': has_disagreement,
            'classification_disagreement': class_disagreement,
            'confidence_disagreement': conf_disagreement,
            'confidence_difference': conf_difference,
            'image_classification': img_class,
            'electrical_classification': elec_class,
            'image_confidence': img_conf,
            'electrical_confidence': elec_conf
        }
    
    def _combine_anomalies(
        self,
        image_result: Optional[Dict],
        electrical_result: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Combine anomalies from different modalities"""
        
        combined_anomalies = []
        
        # Add image anomalies
        if image_result and 'anomalies' in image_result:
            for anomaly in image_result['anomalies']:
                anomaly_copy = anomaly.copy()
                anomaly_copy['source'] = 'image'
                combined_anomalies.append(anomaly_copy)
        
        # Add electrical anomalies
        if electrical_result and 'anomalies' in electrical_result:
            for anomaly in electrical_result['anomalies']:
                anomaly_copy = anomaly.copy()
                anomaly_copy['source'] = 'electrical'
                combined_anomalies.append(anomaly_copy)
        
        # Sort by severity (high -> medium -> low)
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        combined_anomalies.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 2))
        
        return combined_anomalies
    
    def _calculate_context_adjustment(self, context: Optional[Dict]) -> float:
        """Calculate confidence adjustment based on context information"""
        
        if not context:
            return 0.0
        
        adjustment = 0.0
        
        # Part number consistency bonus
        if context.get('part_number_verified'):
            adjustment += 0.05
        
        # Manufacturer consistency
        if context.get('manufacturer_verified'):
            adjustment += 0.03
        
        # Date code consistency
        if context.get('date_code_verified'):
            adjustment += 0.02
        
        # Supply chain information
        supply_chain_risk = context.get('supply_chain_risk', 'unknown')
        if supply_chain_risk == 'low':
            adjustment += 0.05
        elif supply_chain_risk == 'high':
            adjustment -= 0.1
        
        # Batch information
        if context.get('batch_verified'):
            adjustment += 0.02
        
        return min(0.15, max(-0.15, adjustment))  # Cap adjustment at ±15%
    
    def _determine_classification(
        self,
        confidence: float,
        disagreement_info: Dict,
        anomalies: List[Dict]
    ) -> str:
        """Determine final classification based on confidence and other factors"""
        
        # Check for high-severity anomalies
        high_severity_count = sum(1 for a in anomalies if a.get('severity') == 'high')
        
        # If there are high-severity anomalies, be more conservative
        if high_severity_count >= 2:
            return 'FAIL'
        elif high_severity_count >= 1 and confidence < 0.8:
            return 'SUSPECT'
        
        # Check for modality disagreement
        if disagreement_info.get('has_disagreement'):
            if confidence > 0.8:
                return 'SUSPECT'  # High confidence but disagreement = suspect
            else:
                return 'FAIL'     # Low confidence with disagreement = fail
        
        # Standard confidence-based classification
        if confidence >= self.confidence_thresholds['pass']:
            return 'PASS'
        elif confidence >= self.confidence_thresholds['suspect']:
            return 'SUSPECT'
        else:
            return 'FAIL'
    
    def _generate_explanation(
        self,
        classification: str,
        confidence: float,
        available_modalities: List[str],
        disagreement_info: Dict,
        anomalies: List[Dict],
        context_adjustment: float
    ) -> str:
        """Generate human-readable explanation of the analysis result"""
        
        explanations = []
        
        # Base classification explanation
        if classification == 'PASS':
            explanations.append(f"Component appears genuine with {confidence:.1%} confidence.")
        elif classification == 'SUSPECT':
            explanations.append(f"Component shows suspicious characteristics with {confidence:.1%} confidence.")
        else:  # FAIL
            explanations.append(f"Component likely counterfeit with {confidence:.1%} confidence.")
        
        # Modality information
        if len(available_modalities) == 2:
            explanations.append("Analysis based on both visual inspection and electrical testing.")
        elif 'image' in available_modalities:
            explanations.append("Analysis based on visual inspection only.")
        elif 'electrical' in available_modalities:
            explanations.append("Analysis based on electrical testing only.")
        
        # Disagreement information
        if disagreement_info.get('has_disagreement'):
            img_class = disagreement_info.get('image_classification')
            elec_class = disagreement_info.get('electrical_classification')
            explanations.append(f"Note: Visual analysis suggests {img_class} while electrical analysis suggests {elec_class}.")
        
        # Context adjustments
        if abs(context_adjustment) > 0.02:
            if context_adjustment > 0:
                explanations.append("Confidence increased based on supporting context information.")
            else:
                explanations.append("Confidence reduced due to context concerns.")
        
        # Anomaly summary
        high_count = sum(1 for a in anomalies if a.get('severity') == 'high')
        medium_count = sum(1 for a in anomalies if a.get('severity') == 'medium')
        
        if high_count > 0:
            explanations.append(f"Detected {high_count} high-severity anomal{'y' if high_count == 1 else 'ies'}.")
        if medium_count > 0:
            explanations.append(f"Detected {medium_count} medium-severity anomal{'y' if medium_count == 1 else 'ies'}.")
        
        # Top anomaly details
        if anomalies:
            top_anomaly = anomalies[0]
            source = top_anomaly.get('source', 'analysis')
            description = top_anomaly.get('description', 'Unknown anomaly')
            explanations.append(f"Primary concern from {source}: {description}")
        
        return ' '.join(explanations)
    
    def _calculate_adaptive_weights(self, available_modalities: List[str]) -> Dict[str, float]:
        """Calculate adaptive weights based on historical performance"""
        
        # This is a simplified adaptive weighting algorithm
        # In production, this could be much more sophisticated
        
        if len(self.analysis_history) < self.config['min_samples_adaptive']:
            return self.fusion_weights.copy()
        
        # Calculate performance metrics for each modality
        modality_performance = {}
        
        for modality in available_modalities:
            correct_predictions = 0
            total_predictions = 0
            
            for record in self.analysis_history[-100:]:  # Last 100 records
                if 'ground_truth' in record and modality in record.get('modality_results', {}):
                    result = record['modality_results'][modality]
                    predicted = result.get('classification')
                    actual = record['ground_truth']
                    
                    if predicted == actual:
                        correct_predictions += 1
                    total_predictions += 1
            
            if total_predictions > 0:
                modality_performance[modality] = correct_predictions / total_predictions
            else:
                modality_performance[modality] = 0.5  # Neutral
        
        # Adjust weights based on performance
        total_performance = sum(modality_performance.values())
        if total_performance > 0:
            adapted_weights = self.fusion_weights.copy()
            for modality in available_modalities:
                performance_ratio = modality_performance[modality] / (total_performance / len(available_modalities))
                adapted_weights[modality] *= performance_ratio
            
            return adapted_weights
        
        return self.fusion_weights.copy()
    
    def _update_analysis_history(self, fusion_result: Dict):
        """Update analysis history for adaptive learning"""
        
        # Keep only relevant information to avoid memory bloat
        history_record = {
            'timestamp': fusion_result['timestamp'],
            'classification': fusion_result['classification'],
            'confidence': fusion_result['confidence'],
            'modality_results': {},
            'disagreement': fusion_result.get('disagreement', {}),
            'anomaly_count': len(fusion_result.get('anomalies', []))
        }
        
        # Store summary of modality results
        for modality, result in fusion_result.get('modality_results', {}).items():
            if isinstance(result, dict) and 'classification' in result:
                history_record['modality_results'][modality] = {
                    'classification': result['classification'],
                    'confidence': result.get('confidence', 0.0)
                }
        
        self.analysis_history.append(history_record)
        
        # Keep history size manageable
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-500:]  # Keep last 500 records
    
    def add_ground_truth(self, analysis_id: str, ground_truth: str):
        """Add ground truth label for adaptive learning"""
        
        for record in reversed(self.analysis_history):
            if record.get('analysis_id') == analysis_id:
                record['ground_truth'] = ground_truth
                logger.info(f"Ground truth '{ground_truth}' added for analysis {analysis_id}")
                break
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the fusion engine"""
        
        if len(self.analysis_history) < 10:
            return {'message': 'Insufficient data for statistics'}
        
        # Overall statistics
        total_analyses = len(self.analysis_history)
        with_ground_truth = sum(1 for r in self.analysis_history if 'ground_truth' in r)
        
        # Classification distribution
        classifications = {}
        for record in self.analysis_history:
            cls = record.get('classification', 'UNKNOWN')
            classifications[cls] = classifications.get(cls, 0) + 1
        
        # Accuracy (where ground truth is available)
        correct_predictions = 0
        for record in self.analysis_history:
            if 'ground_truth' in record:
                if record['classification'] == record['ground_truth']:
                    correct_predictions += 1
        
        accuracy = correct_predictions / with_ground_truth if with_ground_truth > 0 else None
        
        # Disagreement statistics
        disagreements = sum(1 for r in self.analysis_history if r.get('disagreement', {}).get('has_disagreement'))
        
        return {
            'total_analyses': total_analyses,
            'analyses_with_ground_truth': with_ground_truth,
            'classification_distribution': classifications,
            'accuracy': accuracy,
            'disagreement_rate': disagreements / total_analyses,
            'average_confidence': np.mean([r['confidence'] for r in self.analysis_history]),
            'current_weights': self.fusion_weights
        }


def main():
    """Example usage of the multimodal fusion engine"""
    
    # Initialize fusion engine
    fusion_engine = MultimodalFusionEngine()
    
    # Mock image analysis result
    image_result = {
        'classification': 'SUSPECT',
        'confidence': 0.65,
        'anomalies': [
            {
                'type': 'marking_inconsistency',
                'severity': 'medium',
                'description': 'Font style differs from expected'
            }
        ]
    }
    
    # Mock electrical analysis result
    electrical_result = {
        'classification': 'PASS',
        'confidence': 0.85,
        'anomalies': [
            {
                'type': 'resistance_deviation',
                'severity': 'low',
                'description': 'Slight resistance variation within tolerance'
            }
        ]
    }
    
    # Mock context
    context = {
        'part_number_verified': True,
        'supply_chain_risk': 'low',
        'manufacturer_verified': False
    }
    
    # Perform fusion analysis
    result = fusion_engine.fuse_analysis_results(
        image_result=image_result,
        electrical_result=electrical_result,
        context=context
    )
    
    print("Fusion Analysis Result:")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Explanation: {result['explanation']}")
    
    if result['disagreement']['has_disagreement']:
        print("\nModality Disagreement Detected:")
        print(f"  Image: {result['disagreement']['image_classification']} ({result['disagreement']['image_confidence']:.3f})")
        print(f"  Electrical: {result['disagreement']['electrical_classification']} ({result['disagreement']['electrical_confidence']:.3f})")
    
    print(f"\nFusion Weights: {result['fusion_weights']}")
    
    if result['anomalies']:
        print("\nCombined Anomalies:")
        for anomaly in result['anomalies']:
            source = anomaly.get('source', 'unknown')
            severity = anomaly.get('severity', 'unknown')
            description = anomaly.get('description', 'No description')
            print(f"  [{source.upper()}] {severity.upper()}: {description}")


if __name__ == "__main__":
    main()