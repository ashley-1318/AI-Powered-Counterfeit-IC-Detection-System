# Database Configuration Guide

This guide covers the database setup and configuration for the CircuitCheck system.

## Supported Databases

CircuitCheck supports two database systems:

- **PostgreSQL** (Recommended for production)
- **SQLite** (Suitable for development and testing)

## Environment Variables

Configure your database connection using these environment variables:

### PostgreSQL Configuration

```bash
DATABASE_URL=postgresql://username:password@hostname:port/database_name
# OR use individual components:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=circuitcheck
DB_USER=postgres
DB_PASSWORD=your_password
```

### SQLite Configuration (Development)

```bash
USE_SQLITE=true
```

### Additional Configuration

```bash
SQLALCHEMY_ECHO=false  # Set to true to log all SQL queries
```

## Initial Setup

### 1. PostgreSQL Setup

#### Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql postgresql-server postgresql-contrib

# macOS (using Homebrew)
brew install postgresql

# Windows
# Download and install from https://www.postgresql.org/download/windows/
```

#### Create Database and User

```sql
-- Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create database user
CREATE USER circuitcheck_user WITH PASSWORD 'your_secure_password';

-- Create database
CREATE DATABASE circuitcheck OWNER circuitcheck_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE circuitcheck TO circuitcheck_user;

-- Exit
\q
```

#### Set Environment Variables

```bash
export DATABASE_URL="postgresql://circuitcheck_user:your_secure_password@localhost:5432/circuitcheck"
```

### 2. SQLite Setup (Development Only)

SQLite requires no separate installation - just set the environment variable:

```bash
export USE_SQLITE=true
```

The database file will be created automatically at `backend/circuitcheck.db`.

## Database Initialization

### Automatic Setup

Run the database setup script to create tables and sample data:

```bash
cd backend
python setup_database.py
```

This script will:

- Create the database if it doesn't exist (PostgreSQL only)
- Create all tables and indexes
- Insert sample data for testing
- Verify the setup

### Manual Setup

If you prefer to set up manually:

```python
from app import create_app
from models import db

app = create_app()
with app.app_context():
    db.create_all()
```

## Database Schema

### Tables

#### users

- `id` (Primary Key)
- `username` (Unique)
- `email` (Unique)
- `password_hash`
- `created_at`
- `is_active`

#### components

- `id` (Primary Key)
- `part_number` (Unique)
- `manufacturer`
- `description`
- `category`
- `package_type`
- `datasheet_url`
- `created_at`

#### test_results

- `id` (Primary Key)
- `user_id` (Foreign Key → users.id)
- `component_id` (Foreign Key → components.id)
- `test_id` (Unique)
- `classification` (PASS/SUSPECT/FAIL)
- `confidence`
- `image_path`
- `analysis_results` (JSON)
- `created_at`

#### electrical_measurements

- `id` (Primary Key)
- `test_result_id` (Foreign Key → test_results.id)
- `measurement_type`
- `pin_combination`
- `value`
- `unit`
- `created_at`

### Indexes

The following indexes are created for optimal performance:

```sql
CREATE INDEX idx_test_results_user_id ON test_results(user_id);
CREATE INDEX idx_test_results_created_at ON test_results(created_at);
CREATE INDEX idx_test_results_classification ON test_results(classification);
CREATE INDEX idx_components_part_number ON components(part_number);
CREATE INDEX idx_electrical_measurements_test_id ON electrical_measurements(test_result_id);
```

## Database Utilities

### Connection Testing

```python
from database_config import test_database_connection

if test_database_connection():
    print("Database connection successful")
else:
    print("Database connection failed")
```

### Getting Database Statistics

```python
from database_config import get_database_stats

stats = get_database_stats()
print(f"Database type: {stats['info']['type']}")
print(f"Total size: {stats['info']['size']}")

for table, table_stats in stats['tables'].items():
    print(f"{table}: {table_stats['row_count']} rows")
```

### Database Maintenance

```python
from database_config import DatabaseUtilities, db_config

utils = DatabaseUtilities(db_config)

# Optimize database performance
utils.optimize_database()

# Clean up old data (older than 90 days)
deleted_count = utils.cleanup_old_data(days_to_keep=90)

# Export table data
utils.export_data('backup.json', 'test_results', format='json')
```

## Production Configuration

### PostgreSQL Production Settings

1. **Connection Pooling**: Already configured with connection pooling (10 connections, max overflow 20)

2. **Performance Settings**: Add these to your PostgreSQL configuration:

```sql
# postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
random_page_cost = 1.1
```

3. **Security**:

- Use strong passwords
- Configure `pg_hba.conf` appropriately
- Enable SSL connections
- Regular security updates

### Environment Variables for Production

```bash
# Use connection pooling and SSL
DATABASE_URL="postgresql://user:pass@host:port/db?sslmode=require&pool_size=20"

# Disable query logging in production
SQLALCHEMY_ECHO=false
```

## Backup and Recovery

### Automated Backups (PostgreSQL)

```bash
#!/bin/bash
# backup_database.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/path/to/backups"
DATABASE="circuitcheck"

pg_dump $DATABASE > "$BACKUP_DIR/circuitcheck_backup_$DATE.sql"
gzip "$BACKUP_DIR/circuitcheck_backup_$DATE.sql"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "circuitcheck_backup_*.sql.gz" -mtime +7 -delete
```

### SQLite Backups

```python
from database_config import db_config

# Create backup
backup_path = db_config.create_backup()
print(f"Backup created: {backup_path}")
```

## Troubleshooting

### Common Issues

#### "Database does not exist"

```bash
# Create the database manually
sudo -u postgres createdb circuitcheck
```

#### "Permission denied" for PostgreSQL

```bash
# Check PostgreSQL service status
sudo systemctl status postgresql

# Restart if needed
sudo systemctl restart postgresql

# Check pg_hba.conf for authentication settings
sudo nano /etc/postgresql/*/main/pg_hba.conf
```

#### "Connection refused"

```bash
# Check if PostgreSQL is listening on the correct port
sudo netstat -plunt | grep postgres

# Check PostgreSQL configuration
sudo nano /etc/postgresql/*/main/postgresql.conf
```

#### "Too many connections"

- Increase max_connections in postgresql.conf
- Check for connection leaks in application code
- Monitor connection pool usage

### Performance Issues

#### Slow Queries

- Enable query logging: `SQLALCHEMY_ECHO=true`
- Use `EXPLAIN ANALYZE` to analyze query performance
- Consider adding indexes for frequently queried columns

#### Large Database Size

- Run `VACUUM` and `ANALYZE` regularly (automated in SQLite)
- Consider archiving old test results
- Monitor disk space usage

### Migration Between Databases

#### SQLite to PostgreSQL

```python
# Export from SQLite
from database_config import DatabaseUtilities, DatabaseConfig

sqlite_config = DatabaseConfig('sqlite:///circuitcheck.db')
sqlite_utils = DatabaseUtilities(sqlite_config)

# Export all tables
for table in ['users', 'components', 'test_results', 'electrical_measurements']:
    sqlite_utils.export_data(f'{table}.json', table, 'json')

# Then import into PostgreSQL using the setup script with the JSON files
```

## Monitoring

### Database Health Checks

```python
from database_config import db_config, DatabaseUtilities

# Regular health check
def database_health_check():
    if not db_config.test_connection():
        return {"status": "unhealthy", "message": "Cannot connect to database"}

    utils = DatabaseUtilities(db_config)
    stats = utils.get_table_statistics()

    # Check for any table errors
    for table, table_stats in stats.items():
        if 'error' in table_stats:
            return {"status": "unhealthy", "message": f"Error in table {table}: {table_stats['error']}"}

    return {"status": "healthy", "tables": len(stats)}

# Run periodically
health = database_health_check()
print(f"Database health: {health['status']}")
```

### Monitoring Queries

Add this to your application startup to monitor slow queries:

```python
import logging
import time
from sqlalchemy import event

# Log slow queries
@event.listens_for(db.engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(db.engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    if total > 1.0:  # Log queries taking more than 1 second
        logging.warning(f"Slow query ({total:.2f}s): {statement}")
```
