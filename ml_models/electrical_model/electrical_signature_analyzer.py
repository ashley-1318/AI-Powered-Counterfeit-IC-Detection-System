"""
Electrical Signature Analysis Model for CircuitCheck
Analyzes electrical measurements to detect counterfeit components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
from datetime import datetime
import json

# Scikit-learn imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElectricalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract meaningful features from electrical measurements"""
    
    def __init__(self):
        self.feature_names = []
        self.reference_values = {}
        
    def fit(self, X, y=None):
        """Learn reference values from training data"""
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Calculate reference statistics
        for column in X.columns:
            if X[column].dtype in ['int64', 'float64']:
                self.reference_values[column] = {
                    'mean': X[column].mean(),
                    'std': X[column].std(),
                    'median': X[column].median(),
                    'q25': X[column].quantile(0.25),
                    'q75': X[column].quantile(0.75)
                }
        
        return self
    
    def transform(self, X):
        """Transform electrical measurements into features"""
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        features = []
        
        for _, row in X.iterrows():
            feature_vector = self._extract_features(row.to_dict())
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_features(self, measurements: Dict) -> List[float]:
        """Extract features from a single measurement"""
        features = []
        
        # Basic measurements
        resistance = measurements.get('resistance', {})
        capacitance = measurements.get('capacitance', {})
        leakage_current = measurements.get('leakage_current', {})
        timing = measurements.get('timing', {})
        
        # Resistance features
        if resistance:
            res_values = list(resistance.values())
            features.extend([
                np.mean(res_values),
                np.std(res_values),
                np.min(res_values),
                np.max(res_values),
                np.ptp(res_values)  # peak-to-peak
            ])
            
            # Ratios between different resistance measurements
            if len(res_values) > 1:
                features.append(max(res_values) / min(res_values))
            else:
                features.append(1.0)
        else:
            features.extend([0.0] * 6)
        
        # Capacitance features
        if capacitance:
            cap_values = list(capacitance.values())
            features.extend([
                np.mean(cap_values),
                np.std(cap_values),
                np.min(cap_values),
                np.max(cap_values)
            ])
        else:
            features.extend([0.0] * 4)
        
        # Leakage current features
        if leakage_current:
            leak_values = list(leakage_current.values())
            features.extend([
                np.mean(leak_values),
                np.std(leak_values),
                np.max(leak_values),
                np.sum(leak_values)
            ])
        else:
            features.extend([0.0] * 4)
        
        # Timing features
        if timing:
            timing_values = list(timing.values())
            features.extend([
                np.mean(timing_values),
                np.std(timing_values),
                timing.get('rise_time', 0.0),
                timing.get('fall_time', 0.0),
                timing.get('propagation_delay', 0.0)
            ])
            
            # Timing ratios
            rise_time = timing.get('rise_time', 1.0)
            fall_time = timing.get('fall_time', 1.0)
            features.append(rise_time / fall_time if fall_time > 0 else 1.0)
        else:
            features.extend([0.0] * 6)
        
        # Cross-domain features
        # Power-related estimates
        if resistance and leakage_current:
            avg_resistance = np.mean(list(resistance.values()))
            max_leakage = np.max(list(leakage_current.values()))
            estimated_power = (max_leakage ** 2) * avg_resistance
            features.append(estimated_power)
        else:
            features.append(0.0)
        
        # Quality factor estimates
        if capacitance and resistance:
            avg_cap = np.mean(list(capacitance.values()))
            avg_res = np.mean(list(resistance.values()))
            # Simplified Q factor estimate
            q_factor = 1.0 / (avg_res * avg_cap * 1e-12) if avg_res > 0 and avg_cap > 0 else 0.0
            features.append(q_factor)
        else:
            features.append(0.0)
        
        return features


class ElectricalSignatureAnalyzer:
    """
    Main class for electrical signature analysis of electronic components
    Uses multiple ML approaches for robust counterfeit detection
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.feature_extractor = ElectricalFeatureExtractor()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expected fraction of anomalies
            random_state=42,
            n_estimators=100
        )
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        )
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        
        # Reference database for component types
        self.component_references = {}
        self.is_trained = False
        
    def load_reference_database(self, reference_file: str):
        """Load reference electrical characteristics for different component types"""
        try:
            with open(reference_file, 'r') as f:
                self.component_references = json.load(f)
            logger.info(f"Loaded reference database with {len(self.component_references)} component types")
        except Exception as e:
            logger.warning(f"Could not load reference database: {e}")
            # Create default references
            self._create_default_references()
    
    def _create_default_references(self):
        """Create default reference values for common component types"""
        self.component_references = {
            'MC74HC00AN': {  # NAND gate
                'resistance': {'pin1_pin14': 1e6, 'pin7_pin14': 0.5},
                'capacitance': {'pin1_gnd': 5.0, 'pin14_gnd': 10.0},
                'leakage_current': {'max': 0.001},
                'timing': {'propagation_delay': 9.0, 'rise_time': 6.0, 'fall_time': 6.0}
            },
            'LM358': {  # Op-amp
                'resistance': {'pin1_pin8': 1e6, 'pin4_pin8': 0.5},
                'capacitance': {'input': 3.0, 'output': 5.0},
                'leakage_current': {'max': 0.045},
                'timing': {'slew_rate': 0.5}
            }
        }
    
    def train(self, training_data: List[Dict], labels: List[str], component_types: List[str]):
        """
        Train the electrical signature model
        
        Args:
            training_data: List of electrical measurement dictionaries
            labels: List of labels ('genuine', 'counterfeit')
            component_types: List of component part numbers
        """
        logger.info(f"Training electrical signature model with {len(training_data)} samples")
        
        # Extract features
        self.feature_extractor.fit(training_data)
        features = self.feature_extractor.transform(training_data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Convert labels to binary (0: genuine, 1: counterfeit)
        y_binary = [1 if label == 'counterfeit' else 0 for label in labels]
        
        # Train anomaly detector (unsupervised)
        self.anomaly_detector.fit(features_pca)
        
        # Train classifier (supervised)
        self.classifier.fit(features_pca, y_binary)
        
        # Evaluate model
        scores = cross_val_score(self.classifier, features_pca, y_binary, cv=5)
        logger.info(f"Cross-validation accuracy: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
        
        self.is_trained = True
        
        if self.model_path:
            self.save_model()
    
    def analyze_component(
        self,
        measurements: Dict[str, Any],
        part_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze electrical measurements for counterfeit detection
        
        Args:
            measurements: Dictionary containing electrical measurements
            part_number: Component part number for reference lookup
            
        Returns:
            Analysis results with classification and anomalies
        """
        if not self.is_trained:
            logger.warning("Model not trained, using rule-based analysis")
            return self._rule_based_analysis(measurements, part_number)
        
        try:
            # Extract features
            features = self.feature_extractor.transform([measurements])
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.decision_function(features_pca)[0]
            is_anomaly = self.anomaly_detector.predict(features_pca)[0] == -1
            
            # Classification
            class_prob = self.classifier.predict_proba(features_pca)[0]
            genuine_prob = class_prob[0]
            counterfeit_prob = class_prob[1]
            
            # Determine classification
            if is_anomaly or counterfeit_prob > 0.7:
                classification = "FAIL"
                confidence = counterfeit_prob
            elif counterfeit_prob > 0.3:
                classification = "SUSPECT"
                confidence = 1 - abs(0.5 - counterfeit_prob) * 2  # Distance from decision boundary
            else:
                classification = "PASS"
                confidence = genuine_prob
            
            # Detect specific anomalies
            anomalies = self._detect_anomalies(measurements, part_number)
            
            return {
                'classification': classification,
                'confidence': float(confidence),
                'anomaly_score': float(anomaly_score),
                'probabilities': {
                    'genuine': float(genuine_prob),
                    'counterfeit': float(counterfeit_prob)
                },
                'anomalies': anomalies,
                'analysis_type': 'ml_based'
            }
            
        except Exception as e:
            logger.error(f"ML analysis failed: {e}, falling back to rule-based")
            return self._rule_based_analysis(measurements, part_number)
    
    def _rule_based_analysis(
        self,
        measurements: Dict[str, Any],
        part_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback rule-based analysis when ML model is not available"""
        
        anomalies = self._detect_anomalies(measurements, part_number)
        
        # Simple scoring based on number and severity of anomalies
        high_severity = sum(1 for a in anomalies if a['severity'] == 'high')
        medium_severity = sum(1 for a in anomalies if a['severity'] == 'medium')
        low_severity = sum(1 for a in anomalies if a['severity'] == 'low')
        
        anomaly_score = high_severity * 3 + medium_severity * 2 + low_severity * 1
        
        if anomaly_score >= 3:
            classification = "FAIL"
            confidence = min(0.95, 0.7 + anomaly_score * 0.05)
        elif anomaly_score >= 1:
            classification = "SUSPECT"
            confidence = 0.6 + anomaly_score * 0.1
        else:
            classification = "PASS"
            confidence = 0.85
        
        return {
            'classification': classification,
            'confidence': confidence,
            'anomaly_score': -anomaly_score,  # Negative for consistency with IsolationForest
            'probabilities': {
                'genuine': confidence if classification == 'PASS' else 1 - confidence,
                'counterfeit': 1 - confidence if classification == 'PASS' else confidence
            },
            'anomalies': anomalies,
            'analysis_type': 'rule_based'
        }
    
    def _detect_anomalies(
        self,
        measurements: Dict[str, Any],
        part_number: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect specific anomalies in electrical measurements"""
        
        anomalies = []
        
        # Get reference values if available
        references = self.component_references.get(part_number, {}) if part_number else {}
        
        # Check resistance anomalies
        resistance = measurements.get('resistance', {})
        ref_resistance = references.get('resistance', {})
        
        for measurement, value in resistance.items():
            ref_value = ref_resistance.get(measurement)
            if ref_value and abs(value - ref_value) / ref_value > 0.2:  # 20% deviation
                severity = 'high' if abs(value - ref_value) / ref_value > 0.5 else 'medium'
                anomalies.append({
                    'type': 'resistance_deviation',
                    'measurement': measurement,
                    'expected': ref_value,
                    'actual': value,
                    'deviation_percent': abs(value - ref_value) / ref_value * 100,
                    'severity': severity,
                    'description': f"Resistance {measurement} deviates by {abs(value - ref_value) / ref_value * 100:.1f}%"
                })
        
        # Check capacitance anomalies
        capacitance = measurements.get('capacitance', {})
        ref_capacitance = references.get('capacitance', {})
        
        for measurement, value in capacitance.items():
            ref_value = ref_capacitance.get(measurement)
            if ref_value and abs(value - ref_value) / ref_value > 0.3:  # 30% deviation
                severity = 'high' if abs(value - ref_value) / ref_value > 0.6 else 'medium'
                anomalies.append({
                    'type': 'capacitance_deviation',
                    'measurement': measurement,
                    'expected': ref_value,
                    'actual': value,
                    'deviation_percent': abs(value - ref_value) / ref_value * 100,
                    'severity': severity,
                    'description': f"Capacitance {measurement} deviates by {abs(value - ref_value) / ref_value * 100:.1f}%"
                })
        
        # Check leakage current anomalies
        leakage = measurements.get('leakage_current', {})
        ref_leakage = references.get('leakage_current', {})
        
        max_leakage = max(leakage.values()) if leakage else 0
        ref_max_leakage = ref_leakage.get('max', 0.01)  # Default 10μA max
        
        if max_leakage > ref_max_leakage * 2:
            severity = 'high' if max_leakage > ref_max_leakage * 5 else 'medium'
            anomalies.append({
                'type': 'excessive_leakage',
                'measurement': 'max_leakage_current',
                'expected': f"< {ref_max_leakage}",
                'actual': max_leakage,
                'severity': severity,
                'description': f"Excessive leakage current: {max_leakage:.3f}μA (expected < {ref_max_leakage:.3f}μA)"
            })
        
        # Check timing anomalies
        timing = measurements.get('timing', {})
        ref_timing = references.get('timing', {})
        
        for measurement, value in timing.items():
            ref_value = ref_timing.get(measurement)
            if ref_value and abs(value - ref_value) / ref_value > 0.4:  # 40% deviation
                severity = 'medium' if abs(value - ref_value) / ref_value > 0.8 else 'low'
                anomalies.append({
                    'type': 'timing_deviation',
                    'measurement': measurement,
                    'expected': ref_value,
                    'actual': value,
                    'deviation_percent': abs(value - ref_value) / ref_value * 100,
                    'severity': severity,
                    'description': f"Timing {measurement} deviates by {abs(value - ref_value) / ref_value * 100:.1f}%"
                })
        
        # Cross-check anomalies (unusual combinations)
        self._detect_cross_parameter_anomalies(measurements, anomalies)
        
        return anomalies
    
    def _detect_cross_parameter_anomalies(
        self,
        measurements: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ):
        """Detect anomalies that involve relationships between different measurements"""
        
        resistance = measurements.get('resistance', {})
        leakage = measurements.get('leakage_current', {})
        timing = measurements.get('timing', {})
        
        # Check if high leakage correlates with low resistance (might indicate damage)
        if resistance and leakage:
            min_resistance = min(resistance.values())
            max_leakage = max(leakage.values())
            
            if min_resistance < 1000 and max_leakage > 0.1:  # Low resistance + high leakage
                anomalies.append({
                    'type': 'resistance_leakage_correlation',
                    'severity': 'high',
                    'description': f"Unusual combination: low resistance ({min_resistance:.0f}Ω) with high leakage ({max_leakage:.3f}μA)"
                })
        
        # Check timing consistency
        if timing:
            rise_time = timing.get('rise_time', 0)
            fall_time = timing.get('fall_time', 0)
            
            if rise_time > 0 and fall_time > 0 and abs(rise_time - fall_time) / min(rise_time, fall_time) > 2:
                anomalies.append({
                    'type': 'timing_asymmetry',
                    'severity': 'medium',
                    'description': f"Unusual timing asymmetry: rise={rise_time:.1f}ns, fall={fall_time:.1f}ns"
                })
    
    def save_model(self):
        """Save the trained model to disk"""
        if not self.model_path:
            logger.warning("No model path specified, cannot save model")
            return
        
        model_data = {
            'feature_extractor': self.feature_extractor,
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector,
            'classifier': self.classifier,
            'pca': self.pca,
            'component_references': self.component_references,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load a previously trained model from disk"""
        if not self.model_path:
            logger.warning("No model path specified, cannot load model")
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_extractor = model_data['feature_extractor']
            self.scaler = model_data['scaler']
            self.anomaly_detector = model_data['anomaly_detector']
            self.classifier = model_data['classifier']
            self.pca = model_data['pca']
            self.component_references = model_data['component_references']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def generate_synthetic_training_data(self, num_samples: int = 1000) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Generate synthetic training data for model development
        This should be replaced with real data in production
        """
        training_data = []
        labels = []
        component_types = []
        
        # Component types to simulate
        components = ['MC74HC00AN', 'LM358', '555_TIMER', 'CD4017', 'LM339']
        
        for i in range(num_samples):
            component = np.random.choice(components)
            is_counterfeit = np.random.random() < 0.3  # 30% counterfeit rate
            
            # Generate base measurements for component type
            if component == 'MC74HC00AN':
                base_resistance = {'pin1_pin14': 1e6, 'pin7_pin14': 0.5}
                base_capacitance = {'pin1_gnd': 5.0, 'pin14_gnd': 10.0}
                base_leakage = {'pin1': 0.001, 'pin14': 0.001}
                base_timing = {'rise_time': 6.0, 'fall_time': 6.0, 'propagation_delay': 9.0}
            else:
                # Generic component
                base_resistance = {'pin1_pin8': 1e6, 'pin4_pin8': 1.0}
                base_capacitance = {'input': 3.0, 'output': 5.0}
                base_leakage = {'pin1': 0.002, 'pin8': 0.003}
                base_timing = {'rise_time': 10.0, 'fall_time': 8.0}
            
            # Add noise and counterfeit variations
            if is_counterfeit:
                # Counterfeit parts often have more variation and different characteristics
                resistance_mult = np.random.normal(1.0, 0.4)  # Higher variation
                capacitance_mult = np.random.normal(1.0, 0.3)
                leakage_mult = np.random.lognormal(0, 0.8)  # Often higher leakage
                timing_mult = np.random.normal(1.0, 0.5)
            else:
                # Genuine parts have tighter tolerances
                resistance_mult = np.random.normal(1.0, 0.1)
                capacitance_mult = np.random.normal(1.0, 0.15)
                leakage_mult = np.random.lognormal(0, 0.3)
                timing_mult = np.random.normal(1.0, 0.2)
            
            measurements = {
                'resistance': {k: v * resistance_mult for k, v in base_resistance.items()},
                'capacitance': {k: v * capacitance_mult for k, v in base_capacitance.items()},
                'leakage_current': {k: v * leakage_mult for k, v in base_leakage.items()},
                'timing': {k: v * timing_mult for k, v in base_timing.items()}
            }
            
            training_data.append(measurements)
            labels.append('counterfeit' if is_counterfeit else 'genuine')
            component_types.append(component)
        
        return training_data, labels, component_types


def main():
    """Example usage of the electrical signature analyzer"""
    
    # Initialize analyzer
    analyzer = ElectricalSignatureAnalyzer(model_path='electrical_signature_model.pkl')
    
    # Create default reference database
    analyzer._create_default_references()
    
    # Generate synthetic training data for demonstration
    logger.info("Generating synthetic training data...")
    training_data, labels, component_types = analyzer.generate_synthetic_training_data(1000)
    
    # Train the model
    analyzer.train(training_data, labels, component_types)
    
    # Test with sample measurements
    test_measurements = {
        'resistance': {
            'pin1_pin14': 950000.0,  # Slightly lower than expected
            'pin7_pin14': 0.6
        },
        'capacitance': {
            'pin1_gnd': 5.2,
            'pin14_gnd': 9.8
        },
        'leakage_current': {
            'pin1': 0.001,
            'pin14': 0.0015
        },
        'timing': {
            'rise_time': 6.5,
            'fall_time': 6.2,
            'propagation_delay': 9.2
        }
    }
    
    # Analyze the component
    result = analyzer.analyze_component(test_measurements, 'MC74HC00AN')
    
    print("Analysis Result:")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"Analysis Type: {result['analysis_type']}")
    
    if result['anomalies']:
        print("\nDetected Anomalies:")
        for anomaly in result['anomalies']:
            print(f"- {anomaly['description']} (Severity: {anomaly['severity']})")
    else:
        print("\nNo anomalies detected")


if __name__ == "__main__":
    main()