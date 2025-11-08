"""
Database configuration and utilities for CircuitCheck
Provides database connection, session management, and utility functions
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from typing import Optional, Generator
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
        self._setup_engine()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or use default"""
        
        # Try to get from environment
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
        
        # Fallback to individual components
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'circuitcheck')
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', 'password')
        
        # Check if we should use SQLite for development
        if os.getenv('USE_SQLITE', 'false').lower() == 'true':
            sqlite_path = os.path.join(os.path.dirname(__file__), 'circuitcheck.db')
            return f'sqlite:///{sqlite_path}'
        
        return f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    
    def _setup_engine(self):
        """Set up database engine with appropriate configuration"""
        
        # Engine configuration
        engine_kwargs = {
            'echo': os.getenv('SQLALCHEMY_ECHO', 'false').lower() == 'true',
            'pool_pre_ping': True,  # Verify connections before use
            'pool_recycle': 3600,   # Recycle connections after 1 hour
        }
        
        # PostgreSQL-specific configuration
        if 'postgresql' in self.database_url:
            engine_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
            })
        
        # SQLite-specific configuration
        elif 'sqlite' in self.database_url:
            engine_kwargs.update({
                'pool_pre_ping': False,  # Not needed for SQLite
                'connect_args': {'check_same_thread': False}
            })
        
        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.session_factory = sessionmaker(bind=self.engine)
        
        # Add connection event listeners
        self._setup_event_listeners()
        
        logger.info(f"Database engine configured for: {self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url}")
    
    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for debugging and optimization"""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time
            # Log slow queries (> 1 second)
            if total > 1.0:
                logger.warning(f"Slow query detected ({total:.2f}s): {statement[:100]}...")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup"""
        
        if self.session_factory is None:
            raise Exception("Database not initialized")
            
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        
        if self.engine is None:
            return False
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> dict:
        """Get information about the database"""
        
        if self.engine is None:
            return {'error': 'Database not initialized'}
        
        try:
            with self.engine.connect() as conn:
                if 'postgresql' in self.database_url:
                    version_result = conn.execute(text("SELECT version()"))
                    version_row = version_result.fetchone()
                    version = version_row[0] if version_row else 'Unknown'
                    
                    # Get database size
                    size_result = conn.execute(text("""
                        SELECT pg_size_pretty(pg_database_size(current_database()))
                    """))
                    size_row = size_result.fetchone()
                    size = size_row[0] if size_row else 'Unknown'
                    
                    pool_info = {}
                    if hasattr(self.engine, 'pool') and self.engine.pool:
                        try:
                            pool_info = {
                                'connection_pool_size': getattr(self.engine.pool, 'size', lambda: 'Unknown')(),
                                'checked_out_connections': getattr(self.engine.pool, 'checkedout', lambda: 'Unknown')()
                            }
                        except:
                            pool_info = {'pool_info': 'unavailable'}
                    
                    return {
                        'type': 'PostgreSQL',
                        'version': version.split()[1] if isinstance(version, str) else version,
                        'size': size,
                        **pool_info
                    }
                
                elif 'sqlite' in self.database_url:
                    version_result = conn.execute(text("SELECT sqlite_version()"))
                    version_row = version_result.fetchone()
                    version = version_row[0] if version_row else 'Unknown'
                    
                    # Get database file size
                    db_path = self.database_url.replace('sqlite:///', '')
                    size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
                    size_mb = round(size / (1024 * 1024), 2)
                    
                    return {
                        'type': 'SQLite',
                        'version': version,
                        'size': f'{size_mb} MB',
                        'file_path': db_path
                    }
                
                else:
                    return {'error': 'Unsupported database type'}
                
        except Exception as e:
            logger.error(f"Could not get database info: {e}")
            return {'error': str(e)}
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """Create a database backup (SQLite only for now)"""
        
        if 'sqlite' not in self.database_url:
            raise NotImplementedError("Backup currently only supported for SQLite")
        
        import shutil
        from datetime import datetime
        
        # Source database file
        source_path = self.database_url.replace('sqlite:///', '')
        
        # Backup file path
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(os.path.dirname(source_path), 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f'circuitcheck_backup_{timestamp}.db')
        
        # Create backup
        shutil.copy2(source_path, backup_path)
        logger.info(f"Database backup created: {backup_path}")
        
        return backup_path


class DatabaseUtilities:
    """Database utility functions and maintenance operations"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def get_table_statistics(self) -> dict:
        """Get statistics about database tables"""
        
        stats = {}
        
        try:
            with self.db_config.get_session() as session:
                # Get row counts for each table
                tables = ['users', 'components', 'test_results', 'electrical_measurements']
                
                for table in tables:
                    try:
                        if 'postgresql' in self.db_config.database_url:
                            result = session.execute(text(f"""
                                SELECT 
                                    COUNT(*) as row_count,
                                    pg_size_pretty(pg_total_relation_size('{table}')) as size
                                FROM {table}
                            """))
                            row = result.fetchone()
                            if row:
                                stats[table] = {
                                    'row_count': row[0],
                                    'size': row[1]
                                }
                            else:
                                stats[table] = {'row_count': 0, 'size': 'N/A'}
                        else:  # SQLite
                            result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                            row_data = result.fetchone()
                            row_count = row_data[0] if row_data else 0
                            stats[table] = {
                                'row_count': row_count,
                                'size': 'N/A'
                            }
                    except Exception as e:
                        stats[table] = {'error': str(e)}
                        
        except Exception as e:
            logger.error(f"Could not get table statistics: {e}")
            return {'error': str(e)}
        
        return stats
    
    def optimize_database(self):
        """Perform database optimization operations"""
        
        try:
            with self.db_config.get_session() as session:
                if 'postgresql' in self.db_config.database_url:
                    # Analyze tables for query planner
                    tables = ['users', 'components', 'test_results', 'electrical_measurements']
                    for table in tables:
                        session.execute(text(f"ANALYZE {table}"))
                    logger.info("PostgreSQL table analysis completed")
                    
                elif 'sqlite' in self.db_config.database_url:
                    # Vacuum and analyze for SQLite
                    session.execute(text("VACUUM"))
                    session.execute(text("ANALYZE"))
                    logger.info("SQLite vacuum and analysis completed")
                    
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old test results and related data"""
        
        try:
            with self.db_config.get_session() as session:
                # Delete old test results (and cascade to electrical measurements)
                cutoff_date = f"CURRENT_TIMESTAMP - INTERVAL '{days_to_keep} days'"
                if 'sqlite' in self.db_config.database_url:
                    cutoff_date = f"datetime('now', '-{days_to_keep} days')"
                
                result = session.execute(text(f"""
                    DELETE FROM test_results 
                    WHERE created_at < {cutoff_date}
                """))
                
                deleted_count = getattr(result, 'rowcount', 0) or 0
                logger.info(f"Cleaned up {deleted_count} old test results")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return 0
    
    def export_data(self, export_path: str, table_name: str, format: str = 'json'):
        """Export table data to file"""
        
        try:
            with self.db_config.get_session() as session:
                if table_name not in ['users', 'components', 'test_results', 'electrical_measurements']:
                    raise ValueError(f"Invalid table name: {table_name}")
                
                # Get all data from table
                result = session.execute(text(f"SELECT * FROM {table_name}"))
                rows = result.fetchall()
                columns = result.keys()
                
                if format.lower() == 'json':
                    import json
                    data = [dict(zip(columns, row)) for row in rows]
                    # Convert datetime objects to strings
                    for row in data:
                        for key, value in row.items():
                            if hasattr(value, 'isoformat'):
                                row[key] = value.isoformat()
                    
                    with open(export_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                
                elif format.lower() == 'csv':
                    import csv
                    with open(export_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(columns)
                        writer.writerows(rows)
                
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                logger.info(f"Exported {len(rows)} rows from {table_name} to {export_path}")
                return len(rows)
                
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return 0


# Global database configuration instance
db_config = DatabaseConfig()

# Convenience functions
def get_db_session():
    """Get a database session (for use with context manager)"""
    return db_config.get_session()

def test_database_connection():
    """Test database connection"""
    return db_config.test_connection()

def get_database_stats():
    """Get database statistics"""
    utils = DatabaseUtilities(db_config)
    return {
        'info': db_config.get_database_info(),
        'tables': utils.get_table_statistics()
    }


def main():
    """Test database configuration and utilities"""
    
    print("Testing database configuration...")
    
    # Test connection
    if db_config.test_connection():
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")
        return
    
    # Get database info
    db_info = db_config.get_database_info()
    print(f"Database Type: {db_info.get('type', 'Unknown')}")
    print(f"Version: {db_info.get('version', 'Unknown')}")
    print(f"Size: {db_info.get('size', 'Unknown')}")
    
    # Get table statistics
    utils = DatabaseUtilities(db_config)
    table_stats = utils.get_table_statistics()
    
    print("\nTable Statistics:")
    for table, stats in table_stats.items():
        if 'error' in stats:
            print(f"  {table}: Error - {stats['error']}")
        else:
            print(f"  {table}: {stats['row_count']} rows")
    
    print("\nDatabase configuration test completed!")


if __name__ == "__main__":
    main()