from database.db import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, admin, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    test_results = db.relationship('TestResult', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Component(db.Model):
    """Electronic component model"""
    __tablename__ = 'components'
    
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.String(100), nullable=False)
    manufacturer = db.Column(db.String(100))
    description = db.Column(db.Text)
    category = db.Column(db.String(50))  # Logic IC, Analog IC, etc.
    package_type = db.Column(db.String(50))  # DIP-14, DIP-8, etc.
    reference_data = db.Column(JSON)  # Store reference electrical measurements
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    test_results = db.relationship('TestResult', backref='component', lazy=True)
    
    def __repr__(self):
        return f'<Component {self.part_number}>'

class TestResult(db.Model):
    """Component test result model"""
    __tablename__ = 'test_results'
    
    id = db.Column(db.Integer, primary_key=True)
    component_id = db.Column(db.Integer, db.ForeignKey('components.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    test_id = db.Column(db.String(100), unique=True, nullable=False)
    test_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Classification and confidence
    classification = db.Column(db.String(10))  # PASS, SUSPECT, FAIL
    confidence = db.Column(db.Float)
    
    # Paths to saved images
    image_path = db.Column(db.String(255))
    macro_image_path = db.Column(db.String(255))
    micro_image_path = db.Column(db.String(255))
    
    # Analysis results
    analysis_results = db.Column(JSON)
    image_score = db.Column(db.Float)
    electrical_score = db.Column(db.Float)
    fusion_score = db.Column(db.Float)
    result_class = db.Column(db.String(10))  # PASS, SUSPECT, FAIL (legacy field)
    anomaly_data = db.Column(JSON)  # Store anomaly locations and descriptions
    
    # Electrical measurements
    electrical_measurements = db.Column(JSON)
    
    # Additional metadata
    notes = db.Column(db.Text)
    batch_id = db.Column(db.String(50))  # For grouping related tests
    
    # Relationships
    electrical_measurements_rel = db.relationship('ElectricalMeasurement', backref='test_result', lazy=True)
    
    def __repr__(self):
        return f'<TestResult {self.test_id} - {self.classification}>'

class ElectricalMeasurement(db.Model):
    """Individual electrical measurement data"""
    __tablename__ = 'electrical_measurements'
    
    id = db.Column(db.Integer, primary_key=True)
    test_result_id = db.Column(db.Integer, db.ForeignKey('test_results.id'), nullable=False)
    measurement_type = db.Column(db.String(50), nullable=False)  # resistance, capacitance, etc.
    pin_combination = db.Column(db.String(50))  # pin1_pin14, etc.
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20))  # Ω, pF, μA, etc.
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ElectricalMeasurement {self.measurement_type}:{self.pin_combination} = {self.value}{self.unit}>'