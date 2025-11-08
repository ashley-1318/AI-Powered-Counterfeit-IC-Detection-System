from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import logging

db = SQLAlchemy()
logger = logging.getLogger(__name__)

def init_db(app):
    """Initialize the database with the Flask app"""
    try:
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URI']
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        db.init_app(app)
        
        with app.app_context():
            # Import models here to ensure they're loaded before creating tables
            from database.models import User, Component, TestResult, ElectricalMeasurement
            
            # Create all tables
            db.create_all()
            
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def get_db_session(db_uri):
    """Create a database session outside of the Flask context"""
    engine = create_engine(db_uri)
    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)