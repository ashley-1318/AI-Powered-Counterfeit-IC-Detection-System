from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import logging
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

from database.db import db
from database.models import User

auth_bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Check for required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
                
        # Check if username or email already exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 409
            
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 409
            
        # Create new user
        new_user = User(
            username=data['username'],
            email=data['email'],
            role=data.get('role', 'user')  # Default to 'user' role
        )
        new_user.set_password(data['password'])
        
        db.session.add(new_user)
        db.session.commit()
        
        logger.info(f"User registered: {new_user.username}")
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': new_user.id
        }), 201
        
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login a user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Check for required fields
        if 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password are required'}), 400
            
        # Get user by username
        user = User.query.filter_by(username=data['username']).first()
        
        # Check if user exists and password is correct
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid username or password'}), 401
            
        # Update last login time
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate JWT token
        expiration = datetime.utcnow() + timedelta(days=1)  # Token expires in 1 day
        token = jwt.encode(
            {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'exp': expiration
            },
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
        
        logger.info(f"User login: {user.username}")
        
        return jsonify({
            'token': token,
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'expires': expiration.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error logging in user: {str(e)}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get user profile (requires authentication)"""
    try:
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Authorization token required'}), 401
            
        token = token.split(' ')[1]
        
        try:
            payload = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        user_id = payload['user_id']
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None
        })
        
    except Exception as e:
        logger.error(f"Error retrieving user profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

def token_required(f):
    """Decorator for routes that require authentication"""
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Authorization token required'}), 401
            
        token = token.split(' ')[1]
        
        try:
            payload = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            
            # Add user_id to kwargs
            kwargs['user_id'] = payload['user_id']
            kwargs['user_role'] = payload['role']
            
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
    return decorated