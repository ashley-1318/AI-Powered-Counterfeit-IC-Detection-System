"""
Data Export Utilities for CircuitCheck
Provides various data export formats and utilities
"""

import csv
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExporter:
    """Handles data export in various formats"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'exports')
        self.ensure_output_directory()
    
    def ensure_output_directory(self):
        """Ensure the output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_test_results(
        self,
        test_results: List[Dict[str, Any]],
        format: str = 'csv',
        filename: Optional[str] = None,
        include_measurements: bool = True,
        date_filter: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Export test results to specified format
        
        Args:
            test_results: List of test result dictionaries
            format: Export format ('csv', 'json', 'xml')
            filename: Custom filename (optional)
            include_measurements: Include electrical measurements
            date_filter: Date range filter {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
            
        Returns:
            Path to exported file
        """
        
        # Apply date filter if specified
        if date_filter:
            test_results = self._filter_by_date(test_results, date_filter)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_results_export_{timestamp}.{format.lower()}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Export based on format
        if format.lower() == 'csv':
            return self._export_csv(test_results, filepath, include_measurements)
        elif format.lower() == 'json':
            return self._export_json(test_results, filepath, include_measurements)
        elif format.lower() == 'xml':
            return self._export_xml(test_results, filepath, include_measurements)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_component_inventory(
        self,
        components: List[Dict[str, Any]],
        format: str = 'csv',
        filename: Optional[str] = None
    ) -> str:
        """Export component inventory data"""
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"component_inventory_{timestamp}.{format.lower()}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if format.lower() == 'csv':
            return self._export_components_csv(components, filepath)
        elif format.lower() == 'json':
            return self._export_components_json(components, filepath)
        else:
            raise ValueError(f"Unsupported format for component export: {format}")
    
    def export_analysis_summary(
        self,
        test_results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Export analysis summary with statistics"""
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_summary_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(test_results)
        
        # Export summary
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Analysis summary exported: {filepath}")
        return filepath
    
    def _filter_by_date(
        self,
        test_results: List[Dict[str, Any]],
        date_filter: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Filter test results by date range"""
        
        start_date = datetime.fromisoformat(date_filter.get('start', '1900-01-01'))
        end_date = datetime.fromisoformat(date_filter.get('end', '2100-12-31'))
        
        filtered_results = []
        for result in test_results:
            test_date_str = result.get('created_at', '')
            if test_date_str:
                try:
                    test_date = datetime.fromisoformat(test_date_str.replace('Z', '+00:00'))
                    if start_date <= test_date <= end_date:
                        filtered_results.append(result)
                except ValueError:
                    # Skip invalid dates
                    continue
        
        logger.info(f"Filtered {len(test_results)} results to {len(filtered_results)} within date range")
        return filtered_results
    
    def _export_csv(
        self,
        test_results: List[Dict[str, Any]],
        filepath: str,
        include_measurements: bool
    ) -> str:
        """Export test results to CSV format"""
        
        if not test_results:
            # Create empty file
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['No data to export'])
            return filepath
        
        # Define CSV headers
        headers = [
            'test_id',
            'part_number',
            'manufacturer',
            'description',
            'category',
            'package_type',
            'classification',
            'confidence',
            'test_date',
            'user_name',
            'image_path',
            'image_analysis_confidence',
            'electrical_analysis_confidence',
            'total_anomalies',
            'high_severity_anomalies',
            'medium_severity_anomalies',
            'low_severity_anomalies'
        ]
        
        # Add measurement headers if requested
        if include_measurements:
            measurement_headers = [
                'resistance_measurements',
                'capacitance_measurements', 
                'leakage_current_measurements',
                'timing_measurements'
            ]
            headers.extend(measurement_headers)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for result in test_results:
                # Extract analysis results
                analysis_results = result.get('analysis_results', {})
                img_analysis = analysis_results.get('image_analysis', {})
                elec_analysis = analysis_results.get('electrical_analysis', {})
                
                # Count anomalies by severity
                all_anomalies = []
                all_anomalies.extend(img_analysis.get('anomalies', []))
                all_anomalies.extend(elec_analysis.get('anomalies', []))
                
                severity_counts = {'high': 0, 'medium': 0, 'low': 0}
                for anomaly in all_anomalies:
                    severity = anomaly.get('severity', 'low')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Prepare row data
                row_data = {
                    'test_id': result.get('test_id', ''),
                    'part_number': result.get('part_number', ''),
                    'manufacturer': result.get('manufacturer', ''),
                    'description': result.get('description', ''),
                    'category': result.get('category', ''),
                    'package_type': result.get('package_type', ''),
                    'classification': result.get('classification', ''),
                    'confidence': result.get('confidence', 0),
                    'test_date': result.get('created_at', ''),
                    'user_name': result.get('user_name', ''),
                    'image_path': result.get('image_path', ''),
                    'image_analysis_confidence': img_analysis.get('confidence', ''),
                    'electrical_analysis_confidence': elec_analysis.get('confidence', ''),
                    'total_anomalies': len(all_anomalies),
                    'high_severity_anomalies': severity_counts['high'],
                    'medium_severity_anomalies': severity_counts['medium'],
                    'low_severity_anomalies': severity_counts['low']
                }
                
                # Add measurement data if requested
                if include_measurements:
                    measurements = result.get('electrical_measurements', {})
                    row_data.update({
                        'resistance_measurements': json.dumps(measurements.get('resistance', {})),
                        'capacitance_measurements': json.dumps(measurements.get('capacitance', {})),
                        'leakage_current_measurements': json.dumps(measurements.get('leakage_current', {})),
                        'timing_measurements': json.dumps(measurements.get('timing', {}))
                    })
                
                writer.writerow(row_data)
        
        logger.info(f"CSV export completed: {filepath} ({len(test_results)} records)")
        return filepath
    
    def _export_json(
        self,
        test_results: List[Dict[str, Any]],
        filepath: str,
        include_measurements: bool
    ) -> str:
        """Export test results to JSON format"""
        
        # Prepare export data
        export_data = {
            'export_metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_records': len(test_results),
                'include_measurements': include_measurements,
                'export_version': '1.0'
            },
            'summary_statistics': self._calculate_summary_statistics(test_results),
            'test_results': test_results if include_measurements else [
                self._strip_measurements(result) for result in test_results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"JSON export completed: {filepath} ({len(test_results)} records)")
        return filepath
    
    def _export_xml(
        self,
        test_results: List[Dict[str, Any]],
        filepath: str,
        include_measurements: bool
    ) -> str:
        """Export test results to XML format"""
        
        # Create root element
        root = ET.Element('circuitcheck_export')
        root.set('version', '1.0')
        root.set('exported_at', datetime.now().isoformat())
        
        # Add metadata
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata, 'total_records').text = str(len(test_results))
        ET.SubElement(metadata, 'include_measurements').text = str(include_measurements)
        
        # Add test results
        results_element = ET.SubElement(root, 'test_results')
        
        for result in test_results:
            result_element = ET.SubElement(results_element, 'test_result')
            
            # Basic information
            ET.SubElement(result_element, 'test_id').text = result.get('test_id', '')
            ET.SubElement(result_element, 'part_number').text = result.get('part_number', '')
            ET.SubElement(result_element, 'manufacturer').text = result.get('manufacturer', '')
            ET.SubElement(result_element, 'classification').text = result.get('classification', '')
            ET.SubElement(result_element, 'confidence').text = str(result.get('confidence', 0))
            ET.SubElement(result_element, 'test_date').text = result.get('created_at', '')
            
            # Analysis results
            analysis_results = result.get('analysis_results', {})
            if analysis_results:
                analysis_element = ET.SubElement(result_element, 'analysis_results')
                
                # Image analysis
                if 'image_analysis' in analysis_results:
                    img_element = ET.SubElement(analysis_element, 'image_analysis')
                    img_analysis = analysis_results['image_analysis']
                    ET.SubElement(img_element, 'confidence').text = str(img_analysis.get('confidence', 0))
                    
                    # Anomalies
                    anomalies = img_analysis.get('anomalies', [])
                    if anomalies:
                        anomalies_element = ET.SubElement(img_element, 'anomalies')
                        for anomaly in anomalies:
                            anomaly_element = ET.SubElement(anomalies_element, 'anomaly')
                            ET.SubElement(anomaly_element, 'type').text = anomaly.get('type', '')
                            ET.SubElement(anomaly_element, 'severity').text = anomaly.get('severity', '')
                            ET.SubElement(anomaly_element, 'description').text = anomaly.get('description', '')
                
                # Electrical analysis
                if 'electrical_analysis' in analysis_results:
                    elec_element = ET.SubElement(analysis_element, 'electrical_analysis')
                    elec_analysis = analysis_results['electrical_analysis']
                    ET.SubElement(elec_element, 'confidence').text = str(elec_analysis.get('confidence', 0))
                    
                    # Anomalies
                    anomalies = elec_analysis.get('anomalies', [])
                    if anomalies:
                        anomalies_element = ET.SubElement(elec_element, 'anomalies')
                        for anomaly in anomalies:
                            anomaly_element = ET.SubElement(anomalies_element, 'anomaly')
                            ET.SubElement(anomaly_element, 'type').text = anomaly.get('type', '')
                            ET.SubElement(anomaly_element, 'severity').text = anomaly.get('severity', '')
                            ET.SubElement(anomaly_element, 'description').text = anomaly.get('description', '')
            
            # Measurements
            if include_measurements:
                measurements = result.get('electrical_measurements', {})
                if measurements:
                    measurements_element = ET.SubElement(result_element, 'electrical_measurements')
                    
                    for measurement_type, data in measurements.items():
                        type_element = ET.SubElement(measurements_element, measurement_type)
                        if isinstance(data, dict):
                            for pin_combo, value in data.items():
                                measurement = ET.SubElement(type_element, 'measurement')
                                ET.SubElement(measurement, 'pins').text = pin_combo
                                ET.SubElement(measurement, 'value').text = str(value)
        
        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"XML export completed: {filepath} ({len(test_results)} records)")
        return filepath
    
    def _export_components_csv(self, components: List[Dict[str, Any]], filepath: str) -> str:
        """Export component inventory to CSV"""
        
        if not components:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['No components to export'])
            return filepath
        
        headers = [
            'part_number',
            'manufacturer',
            'description',
            'category',
            'package_type',
            'datasheet_url',
            'created_at',
            'total_tests',
            'pass_count',
            'suspect_count',
            'fail_count'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for component in components:
                # Calculate test statistics
                test_stats = component.get('test_statistics', {})
                
                row_data = {
                    'part_number': component.get('part_number', ''),
                    'manufacturer': component.get('manufacturer', ''),
                    'description': component.get('description', ''),
                    'category': component.get('category', ''),
                    'package_type': component.get('package_type', ''),
                    'datasheet_url': component.get('datasheet_url', ''),
                    'created_at': component.get('created_at', ''),
                    'total_tests': test_stats.get('total', 0),
                    'pass_count': test_stats.get('pass', 0),
                    'suspect_count': test_stats.get('suspect', 0),
                    'fail_count': test_stats.get('fail', 0)
                }
                
                writer.writerow(row_data)
        
        logger.info(f"Component inventory CSV exported: {filepath}")
        return filepath
    
    def _export_components_json(self, components: List[Dict[str, Any]], filepath: str) -> str:
        """Export component inventory to JSON"""
        
        export_data = {
            'export_metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_components': len(components),
                'export_type': 'component_inventory'
            },
            'components': components
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Component inventory JSON exported: {filepath}")
        return filepath
    
    def _strip_measurements(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove measurement data from result to reduce export size"""
        
        stripped_result = result.copy()
        if 'electrical_measurements' in stripped_result:
            del stripped_result['electrical_measurements']
        
        return stripped_result
    
    def _calculate_summary_statistics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for test results"""
        
        if not test_results:
            return {
                'total_tests': 0,
                'classification_counts': {},
                'average_confidence': 0,
                'date_range': {}
            }
        
        # Classification counts
        classifications = [r.get('classification', 'UNKNOWN') for r in test_results]
        classification_counts = {}
        for cls in ['PASS', 'SUSPECT', 'FAIL', 'UNKNOWN']:
            classification_counts[cls] = classifications.count(cls)
        
        # Confidence statistics
        confidences = [r.get('confidence', 0) for r in test_results if r.get('confidence') is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Date range
        dates = [r.get('created_at', '') for r in test_results if r.get('created_at')]
        date_range = {}
        if dates:
            date_range = {
                'earliest': min(dates),
                'latest': max(dates)
            }
        
        # Component type statistics
        components = [r.get('part_number', 'Unknown') for r in test_results]
        unique_components = len(set(components))
        
        # User statistics
        users = [r.get('user_name', 'Unknown') for r in test_results]
        unique_users = len(set(users))
        
        # Anomaly statistics
        total_anomalies = 0
        anomaly_types = {}
        
        for result in test_results:
            analysis_results = result.get('analysis_results', {})
            
            # Count image anomalies
            img_anomalies = analysis_results.get('image_analysis', {}).get('anomalies', [])
            total_anomalies += len(img_anomalies)
            
            for anomaly in img_anomalies:
                anomaly_type = anomaly.get('type', 'unknown')
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            # Count electrical anomalies
            elec_anomalies = analysis_results.get('electrical_analysis', {}).get('anomalies', [])
            total_anomalies += len(elec_anomalies)
            
            for anomaly in elec_anomalies:
                anomaly_type = anomaly.get('type', 'unknown')
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
        
        return {
            'total_tests': len(test_results),
            'classification_counts': classification_counts,
            'pass_rate': classification_counts.get('PASS', 0) / len(test_results),
            'average_confidence': avg_confidence,
            'confidence_range': {
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            },
            'date_range': date_range,
            'unique_components': unique_components,
            'unique_users': unique_users,
            'total_anomalies': total_anomalies,
            'anomaly_types': anomaly_types
        }


def main():
    """Example usage of data export utilities"""
    
    # Create sample test results
    sample_results = [
        {
            'test_id': 'TEST_001',
            'part_number': 'MC74HC00AN',
            'manufacturer': 'ON Semiconductor',
            'description': 'Quad NAND Gate',
            'category': 'Logic IC',
            'package_type': 'DIP-14',
            'classification': 'PASS',
            'confidence': 0.92,
            'created_at': '2024-01-15T10:30:00Z',
            'user_name': 'engineer1',
            'image_path': '/uploads/test001.jpg',
            'analysis_results': {
                'image_analysis': {
                    'confidence': 0.89,
                    'anomalies': []
                },
                'electrical_analysis': {
                    'confidence': 0.95,
                    'anomalies': []
                }
            },
            'electrical_measurements': {
                'resistance': {'pin1_pin14': 1000000, 'pin7_pin14': 0.5},
                'capacitance': {'pin1_gnd': 5.0, 'pin14_gnd': 10.0}
            }
        },
        {
            'test_id': 'TEST_002',
            'part_number': 'LM358N',
            'manufacturer': 'Texas Instruments',
            'description': 'Dual Op Amp',
            'category': 'Analog IC',
            'package_type': 'DIP-8',
            'classification': 'FAIL',
            'confidence': 0.88,
            'created_at': '2024-01-16T14:20:00Z',
            'user_name': 'engineer2',
            'image_path': '/uploads/test002.jpg',
            'analysis_results': {
                'image_analysis': {
                    'confidence': 0.82,
                    'anomalies': [
                        {'type': 'marking_inconsistency', 'severity': 'high', 'description': 'Logo altered'}
                    ]
                },
                'electrical_analysis': {
                    'confidence': 0.93,
                    'anomalies': [
                        {'type': 'resistance_deviation', 'severity': 'medium', 'description': 'High resistance'}
                    ]
                }
            },
            'electrical_measurements': {
                'resistance': {'pin1_pin8': 1200000, 'pin4_pin8': 0.65},
                'leakage_current': {'pin1': 0.015, 'pin8': 0.018}
            }
        }
    ]
    
    # Initialize exporter
    exporter = DataExporter()
    
    print("Testing data export utilities...")
    
    # Test CSV export
    try:
        csv_path = exporter.export_test_results(
            sample_results,
            format='csv',
            include_measurements=True
        )
        print(f"CSV export successful: {csv_path}")
    except Exception as e:
        print(f"CSV export failed: {e}")
    
    # Test JSON export
    try:
        json_path = exporter.export_test_results(
            sample_results,
            format='json',
            include_measurements=True
        )
        print(f"JSON export successful: {json_path}")
    except Exception as e:
        print(f"JSON export failed: {e}")
    
    # Test XML export
    try:
        xml_path = exporter.export_test_results(
            sample_results,
            format='xml',
            include_measurements=True
        )
        print(f"XML export successful: {xml_path}")
    except Exception as e:
        print(f"XML export failed: {e}")
    
    # Test analysis summary
    try:
        summary_path = exporter.export_analysis_summary(sample_results)
        print(f"Analysis summary export successful: {summary_path}")
    except Exception as e:
        print(f"Analysis summary export failed: {e}")
    
    print("Data export testing completed!")


if __name__ == "__main__":
    main()