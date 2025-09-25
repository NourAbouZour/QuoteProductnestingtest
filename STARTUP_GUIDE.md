# DXF Quotation Application - Startup Guide

## Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
python startup_script.py
```

### Option 2: Using the Batch File (Windows)
```bash
start_server.bat
```

### Option 3: Direct Flask Start
```bash
python app.py
```

## What the Startup Script Does

The `startup_script.py` performs the following checks and actions:

1. **Python Version Check** - Ensures Python 3.8+ is installed
2. **Dependency Validation** - Checks all required packages are available
3. **Directory Creation** - Creates necessary folders (uploads, temp, logs, cache, build)
4. **File Permissions** - Verifies write permissions
5. **Database Connection** - Tests database connectivity
6. **Application Startup** - Starts the Flask server on http://0.0.0.0:5000

## Server Access

Once started, the application will be available at:
- **Local Access**: http://localhost:5000
- **Network Access**: http://[your-ip]:5000

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database Connection Issues**
   - Check DatabaseConfig.py settings
   - Ensure database server is running
   - Verify credentials

3. **Port Already in Use**
   - Check if another instance is running
   - Use `netstat -an | findstr :5000` to check port usage
   - Kill existing processes if needed

4. **Permission Errors**
   - Run as Administrator on Windows
   - Check file/folder permissions

### Development vs Production

- **Development**: The script runs with `debug=False` for stability
- **Production**: For production deployment, consider using a WSGI server like Gunicorn

## File Structure

```
├── startup_script.py      # Main startup script
├── start_server.bat      # Windows batch file
├── app.py               # Flask application
├── deploy.bat           # Full deployment script
└── requirements.txt     # Python dependencies
```

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure database configuration is correct
4. Check file permissions and directory access
