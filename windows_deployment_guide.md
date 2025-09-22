# Windows Server 2022 Deployment Guide

## Prerequisites

### 1. System Requirements
- Windows Server 2022
- Python 3.8 or higher
- PostgreSQL 12 or higher
- Minimum 8GB RAM (16GB recommended)
- Minimum 50GB free disk space
- Network access for package installation

### 2. Software Installation

#### Install Python
```powershell
# Download Python 3.11 from python.org
# Install with "Add Python to PATH" checked
# Verify installation
python --version
pip --version
```

#### Install PostgreSQL
```powershell
# Download PostgreSQL from postgresql.org
# Install with default settings
# Create database and user
psql -U postgres
CREATE DATABASE QuotationDB;
CREATE USER dxf_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE QuotationDB TO dxf_user;
\q
```

#### Install Visual C++ Build Tools (if needed)
```powershell
# Download from Microsoft Visual Studio website
# Install "C++ build tools" workload
```

## Deployment Steps

### 1. Prepare Application Directory
```powershell
# Create application directory
mkdir C:\DXFQuotation
cd C:\DXFQuotation

# Copy all application files to this directory
# Ensure the following files are present:
# - app.py
# - requirements.txt
# - All Python modules
# - templates/ folder
# - assets/ folder
```

### 2. Install Dependencies
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional production dependencies
pip install psutil gunicorn
```

### 3. Configure Environment
```powershell
# Create environment file
notepad .env
```

Add the following content to `.env`:
```
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=QuotationDB
DB_USER=dxf_user
DB_PASSWORD=your_secure_password

# Security
SECRET_KEY=your_very_secure_secret_key_here

# Server Configuration
PORT=5000
HOST=0.0.0.0

# Optional: Custom paths
UPLOAD_FOLDER=uploads
TEMP_FOLDER=temp
LOG_FOLDER=logs
```

### 4. Create Required Directories
```powershell
# Create necessary directories
mkdir uploads
mkdir temp
mkdir logs
mkdir cache
```

### 5. Test Configuration
```powershell
# Test the application
python startup_script.py
```

### 6. Create Windows Service (Optional)
```powershell
# Install NSSM (Non-Sucking Service Manager)
# Download from nssm.cc
# Extract to C:\nssm

# Create service
C:\nssm\win64\nssm.exe install DXFQuotation
# Set Application Path: C:\DXFQuotation\venv\Scripts\python.exe
# Set Arguments: C:\DXFQuotation\startup_script.py
# Set Working Directory: C:\DXFQuotation

# Start service
C:\nssm\win64\nssm.exe start DXFQuotation
```

## Production Configuration

### 1. Firewall Configuration
```powershell
# Allow HTTP traffic
New-NetFirewallRule -DisplayName "DXF Quotation HTTP" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow

# Allow HTTPS traffic (if using SSL)
New-NetFirewallRule -DisplayName "DXF Quotation HTTPS" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow
```

### 2. IIS Reverse Proxy (Optional)
```powershell
# Install IIS and URL Rewrite module
# Configure reverse proxy to forward requests to localhost:5000
```

### 3. SSL Certificate (Recommended)
```powershell
# Install SSL certificate for HTTPS
# Configure application to use HTTPS
```

## Monitoring and Maintenance

### 1. Health Checks
- Access `http://your-server:5000/health` for health status
- Access `http://your-server:5000/metrics` for detailed metrics

### 2. Log Monitoring
```powershell
# Monitor application logs
Get-Content C:\DXFQuotation\app.log -Wait -Tail 50
```

### 3. Performance Monitoring
```powershell
# Monitor system resources
Get-Process python | Select-Object ProcessName, CPU, WorkingSet
```

### 4. Database Maintenance
```powershell
# Regular database maintenance
psql -U dxf_user -d QuotationDB -c "VACUUM ANALYZE;"
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```powershell
# Check memory usage
Get-Process python | Select-Object ProcessName, WorkingSet
# Restart application if memory usage is too high
```

#### 2. Database Connection Issues
```powershell
# Test database connection
psql -U dxf_user -d QuotationDB -c "SELECT 1;"
```

#### 3. File Permission Issues
```powershell
# Check directory permissions
icacls C:\DXFQuotation\uploads
icacls C:\DXFQuotation\temp
```

#### 4. Port Already in Use
```powershell
# Check what's using port 5000
netstat -ano | findstr :5000
# Kill process if needed
taskkill /PID <process_id> /F
```

## Security Considerations

### 1. Database Security
- Use strong passwords
- Limit database user permissions
- Enable SSL for database connections

### 2. Application Security
- Use strong secret keys
- Enable HTTPS in production
- Regular security updates

### 3. File System Security
- Restrict upload directory permissions
- Regular cleanup of temporary files
- Monitor disk usage

## Backup Strategy

### 1. Database Backup
```powershell
# Create backup script
pg_dump -U dxf_user -d QuotationDB > backup_$(Get-Date -Format "yyyyMMdd").sql
```

### 2. Application Backup
```powershell
# Backup application files
robocopy C:\DXFQuotation C:\Backup\DXFQuotation /E /R:3 /W:10
```

### 3. Automated Backup
```powershell
# Create scheduled task for daily backups
# Use Windows Task Scheduler
```

## Performance Optimization

### 1. System Optimization
- Enable Windows Performance Toolkit
- Monitor CPU and memory usage
- Optimize database queries

### 2. Application Optimization
- Monitor request response times
- Optimize image processing
- Implement caching where appropriate

### 3. Database Optimization
- Regular VACUUM and ANALYZE
- Monitor slow queries
- Optimize indexes

## Support and Maintenance

### 1. Regular Maintenance Tasks
- Daily: Check logs and health status
- Weekly: Database maintenance and cleanup
- Monthly: Security updates and performance review

### 2. Monitoring Alerts
- Set up alerts for high memory usage
- Monitor error rates
- Track response times

### 3. Update Procedures
- Test updates in staging environment
- Backup before updates
- Document all changes
