from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
import logging

from database.db import init_db
from api.component_analysis import component_analysis_bp
from api.results import results_bp
from api.auth import auth_bp
from api.upload import upload_bp

def create_app(config=None):
    """Factory function to create Flask app"""
    app = Flask(__name__)
    CORS(app)

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
    app.config['DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///circuitcheck.db')
    
    # Override with custom config if provided
    if config:
        app.config.update(config)

    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints
    app.register_blueprint(component_analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(results_bp, url_prefix='/api/results')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(upload_bp, url_prefix='/api/upload')

    # Initialize database
    init_db(app)

    @app.route('/')
    def home():
        return jsonify({'message': 'CircuitCheck API Server is running'})

    @app.route('/health')
    def health_check():
        return jsonify({'status': 'ok'})
        
    return app

# Create app instance for direct execution
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Force debug mode on to get more detailed error information
    app.run(host='0.0.0.0', port=port, debug=True)