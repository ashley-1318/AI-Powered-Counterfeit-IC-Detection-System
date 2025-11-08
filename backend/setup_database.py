"""
Database initialization and migration script for CircuitCheck
Creates tables and sets up the PostgreSQL database schema
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash
from datetime import datetime
import json

# Add the backend directory to the Python path
backend_dir = os.path.dirname(__file__)
sys.path.insert(0, backend_dir)

from app import create_app
from database.models import User, Component, TestResult, ElectricalMeasurement
from database.db import db

def create_database(database_url: str, database_name: str):
    """Create the database if it doesn't exist"""
    
    # Connect to PostgreSQL server (without specifying database)
    server_url = database_url.rsplit('/', 1)[0]
    engine = create_engine(f"{server_url}/postgres")
    
    with engine.connect() as conn:
        # Set autocommit mode
        conn = conn.execution_options(autocommit=True)
        
        # Check if database exists
        result = conn.execute(text(
            "SELECT 1 FROM pg_database WHERE datname = :db_name"
        ), {"db_name": database_name})
        
        if not result.fetchone():
            # Create database
            conn.execute(text(f'CREATE DATABASE "{database_name}"'))
            print(f"Created database: {database_name}")
        else:
            print(f"Database {database_name} already exists")

def init_database(app):
    """Initialize database tables"""
    
    with app.app_context():
        # Drop all tables (for development - remove in production)
        print("Dropping existing tables...")
        db.drop_all()
        
        # Create all tables
        print("Creating tables...")
        db.create_all()
        
        # Create indexes for better performance
        with db.engine.connect() as conn:
            # Index on test results for faster queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_test_results_user_id 
                ON test_results(user_id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_test_results_created_at 
                ON test_results(created_at)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_test_results_classification 
                ON test_results(classification)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_components_part_number 
                ON components(part_number)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_electrical_measurements_test_id 
                ON electrical_measurements(test_result_id)
            """))
            
            # Commit the indexes
            conn.commit()
        
        print("Database tables and indexes created successfully")

def create_sample_data(app):
    """Create sample data for testing and demonstration"""
    
    with app.app_context():
        # Create sample users
        sample_users = [
            {
                'username': 'demo_user',
                'email': 'demo@circuitcheck.com',
                'password': 'demo123'
            },
            {
                'username': 'test_engineer',
                'email': 'engineer@circuitcheck.com', 
                'password': 'test123'
            },
            {
                'username': 'quality_manager',
                'email': 'manager@circuitcheck.com',
                'password': 'manager123'
            }
        ]
        
        created_users = []
        for user_data in sample_users:
            # Check if user already exists
            existing_user = User.query.filter_by(username=user_data['username']).first()
            if not existing_user:
                user = User()
                user.password_hash = generate_password_hash(user_data['password'])
                user.username = user_data['username']
                user.email = user_data['email']
                db.session.add(user)
                created_users.append(user)
                print(f"Created user: {user_data['username']}")
        
        # Create sample components
        sample_components = [
            {
                'part_number': 'MC74HC00AN',
                'manufacturer': 'Motorola/ON Semiconductor',
                'description': 'Quad 2-input NAND gate',
                'category': 'Logic IC',
                'package_type': 'DIP-14'
            },
            {
                'part_number': 'LM358N',
                'manufacturer': 'Texas Instruments',
                'description': 'Dual operational amplifier',
                'category': 'Analog IC',
                'package_type': 'DIP-8'
            },
            {
                'part_number': 'NE555N',
                'manufacturer': 'Texas Instruments',
                'description': 'Timer IC',
                'category': 'Timer IC',
                'package_type': 'DIP-8'
            },
            {
                'part_number': 'CD4017BE',
                'manufacturer': 'Texas Instruments',
                'description': 'Decade counter/divider',
                'category': 'Logic IC',
                'package_type': 'DIP-16'
            },
            {
                'part_number': 'LM339N',
                'manufacturer': 'Texas Instruments',
                'description': 'Quad voltage comparator',
                'category': 'Analog IC',
                'package_type': 'DIP-14'
            }
        ]
        
        created_components = []
        for comp_data in sample_components:
            # Check if component already exists
            existing_comp = Component.query.filter_by(part_number=comp_data['part_number']).first()
            if not existing_comp:
                component = Component()
                component.part_number = comp_data['part_number']
                component.manufacturer = comp_data['manufacturer']
                component.description = comp_data['description']
                component.category = comp_data['category']
                component.package_type = comp_data['package_type']
                db.session.add(component)
                created_components.append(component)
                print(f"Created component: {comp_data['part_number']}")
        
        # Commit users and components first
        db.session.commit()
        
        # Create sample test results
        if created_users and created_components:
            demo_user = created_users[0]
            
            sample_test_results = [
                {
                    'component': created_components[0],  # MC74HC00AN
                    'classification': 'PASS',
                    'confidence': 0.92,
                    'image_path': '/uploads/demo/mc74hc00an_pass.jpg',
                    'analysis_results': {
                        'image_analysis': {
                            'confidence': 0.89,
                            'anomalies': []
                        },
                        'electrical_analysis': {
                            'confidence': 0.95,
                            'anomalies': []
                        },
                        'fusion_analysis': {
                            'explanation': 'Component passes all checks with high confidence'
                        }
                    },
                    'electrical_data': {
                        'resistance': {'pin1_pin14': 980000, 'pin7_pin14': 0.48},
                        'capacitance': {'pin1_gnd': 5.1, 'pin14_gnd': 9.9},
                        'leakage_current': {'pin1': 0.0008, 'pin14': 0.0009},
                        'timing': {'rise_time': 6.2, 'fall_time': 5.8, 'propagation_delay': 8.9}
                    }
                },
                {
                    'component': created_components[0],  # MC74HC00AN
                    'classification': 'FAIL',
                    'confidence': 0.88,
                    'image_path': '/uploads/demo/mc74hc00an_fail.jpg',
                    'analysis_results': {
                        'image_analysis': {
                            'confidence': 0.82,
                            'anomalies': [
                                {'type': 'marking_inconsistency', 'severity': 'high', 'description': 'Logo appears altered'}
                            ]
                        },
                        'electrical_analysis': {
                            'confidence': 0.93,
                            'anomalies': [
                                {'type': 'resistance_deviation', 'severity': 'medium', 'description': 'Pin resistance higher than expected'}
                            ]
                        },
                        'fusion_analysis': {
                            'explanation': 'Visual anomalies combined with electrical deviations indicate counterfeit'
                        }
                    },
                    'electrical_data': {
                        'resistance': {'pin1_pin14': 1250000, 'pin7_pin14': 0.65},
                        'capacitance': {'pin1_gnd': 7.8, 'pin14_gnd': 12.3},
                        'leakage_current': {'pin1': 0.015, 'pin14': 0.018},
                        'timing': {'rise_time': 9.1, 'fall_time': 8.7, 'propagation_delay': 15.2}
                    }
                },
                {
                    'component': created_components[1],  # LM358N
                    'classification': 'SUSPECT',
                    'confidence': 0.65,
                    'image_path': '/uploads/demo/lm358n_suspect.jpg',
                    'analysis_results': {
                        'image_analysis': {
                            'confidence': 0.72,
                            'anomalies': [
                                {'type': 'surface_defect', 'severity': 'medium', 'description': 'Minor surface scratches visible'}
                            ]
                        },
                        'electrical_analysis': {
                            'confidence': 0.58,
                            'anomalies': [
                                {'type': 'timing_deviation', 'severity': 'low', 'description': 'Slew rate slightly outside normal range'}
                            ]
                        },
                        'fusion_analysis': {
                            'explanation': 'Component shows minor anomalies that warrant closer inspection'
                        }
                    },
                    'electrical_data': {
                        'resistance': {'pin1_pin8': 1100000, 'pin4_pin8': 0.52},
                        'capacitance': {'input': 3.2, 'output': 5.3},
                        'leakage_current': {'pin1': 0.048, 'pin8': 0.051},
                        'timing': {'slew_rate': 0.42}
                    }
                }
            ]
            
            for i, test_data in enumerate(sample_test_results, 1):
                test_result = TestResult()
                test_result.user_id = demo_user.id
                test_result.component_id = test_data['component'].id
                test_result.test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
                test_result.classification = test_data['classification']
                test_result.confidence = test_data['confidence']
                test_result.image_path = test_data['image_path']
                test_result.analysis_results = test_data['analysis_results']
                db.session.add(test_result)
                db.session.flush()  # Get the test result ID
                
                # Create electrical measurements
                for measurement_type, measurements in test_data['electrical_data'].items():
                    for pin_combo, value in measurements.items():
                        electrical_measurement = ElectricalMeasurement()
                        electrical_measurement.test_result_id = test_result.id
                        electrical_measurement.measurement_type = measurement_type
                        electrical_measurement.pin_combination = pin_combo
                        electrical_measurement.value = value
                        electrical_measurement.unit = _get_unit_for_measurement_type(measurement_type)
                        db.session.add(electrical_measurement)
                
                print(f"Created test result: {test_result.test_id} ({test_data['classification']})")
        
        # Commit all sample data
        db.session.commit()
        print("Sample data created successfully")

def _get_unit_for_measurement_type(measurement_type: str) -> str:
    """Get the appropriate unit for a measurement type"""
    unit_map = {
        'resistance': 'Ω',
        'capacitance': 'pF', 
        'leakage_current': 'μA',
        'timing': 'ns'
    }
    return unit_map.get(measurement_type, '')

def verify_database_setup(app):
    """Verify that the database setup is working correctly"""
    
    with app.app_context():
        try:
            # Test basic queries
            user_count = User.query.count()
            component_count = Component.query.count()
            test_result_count = TestResult.query.count()
            electrical_measurement_count = ElectricalMeasurement.query.count()
            
            print(f"\nDatabase verification:")
            print(f"Users: {user_count}")
            print(f"Components: {component_count}")
            print(f"Test Results: {test_result_count}")
            print(f"Electrical Measurements: {electrical_measurement_count}")
            
            # Test a complex query
            recent_results = db.session.query(TestResult, Component, User)\
                .join(Component, TestResult.component_id == Component.id)\
                .join(User, TestResult.user_id == User.id)\
                .order_by(TestResult.created_at.desc())\
                .limit(3)\
                .all()
            
            print(f"\nRecent test results:")
            for result_tuple in recent_results:
                test_result, component, user = result_tuple
                print(f"- {component.part_number}: {test_result.classification} ({test_result.confidence:.2f})")
            
            print("\nDatabase setup verification completed successfully!")
            
        except Exception as e:
            print(f"Database verification failed: {e}")
            return False
    
    return True

def main():
    """Main database setup function"""
    
    # Configuration
    database_name = os.getenv('DB_NAME', 'circuitcheck')
    database_url = os.getenv('DATABASE_URL', f'sqlite:///{database_name}.db')
    
    print(f"Setting up database: {database_name}")
    print(f"Database URL: {database_url}")
    
    # Create database if it doesn't exist (only for PostgreSQL)
    if 'postgresql' in database_url:
        try:
            create_database(database_url, database_name)
        except Exception as e:
            print(f"Warning: Could not create database automatically: {e}")
            print("Please ensure PostgreSQL is running and the database exists")
    
    # Create Flask app
    app = create_app()
    
    # Override database URL if provided
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    
    # Initialize database
    try:
        init_database(app)
        
        # Create sample data
        create_sample_data(app)
        
        # Verify setup
        verify_database_setup(app)
        
        print("\n" + "="*50)
        print("DATABASE SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nSample login credentials:")
        print("Username: demo_user, Password: demo123")
        print("Username: test_engineer, Password: test123")
        print("Username: quality_manager, Password: manager123")
        print("\nYou can now start the Flask application.")
        
    except Exception as e:
        print(f"\nDatabase setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)