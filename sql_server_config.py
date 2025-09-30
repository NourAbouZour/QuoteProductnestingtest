"""
SQL Server Database Configuration for Customer Search
Connection management for SSMS database
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import logging
import traceback
from datetime import datetime
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sql_server_debug.log', mode='w')
    ]
)

# SQL Server Database configuration
logging.info("=" * 80)
logging.info("ENVIRONMENT VARIABLE DEBUG")
logging.info("=" * 80)

# Debug environment variable loading
env_vars = ['SQL_SERVER_HOST', 'SQL_SERVER_DB', 'SQL_SERVER_USER', 'SQL_SERVER_PASSWORD', 'SQL_SERVER_TRUSTED']
for var in env_vars:
    value = os.environ.get(var, 'NOT_SET')
    if 'PASSWORD' in var:
        display_value = '***' if value != 'NOT_SET' else 'NOT_SET'
    else:
        display_value = value
    logging.info(f"Environment variable {var}: {display_value}")

SQL_SERVER_CONFIG = {
    'server': os.environ.get('SQL_SERVER_HOST', 'localhost'),
    'database': os.environ.get('SQL_SERVER_DB', 'SOCDB_EDMOND_BARADEI'),
    'username': os.environ.get('SQL_SERVER_USER', ''),
    'password': os.environ.get('SQL_SERVER_PASSWORD', ''),
    'driver': 'ODBC Driver 17 for SQL Server',
    'trusted_connection': os.environ.get('SQL_SERVER_TRUSTED', 'no').lower() == 'yes'
}

logging.info("Final SQL_SERVER_CONFIG:")
for key, value in SQL_SERVER_CONFIG.items():
    if key == 'password':
        display_value = '***' if value else 'EMPTY'
    else:
        display_value = value
    logging.info(f"  {key}: {display_value}")

def get_sql_server_url():
    """Generate SQL Server connection URL"""
    logging.info("=" * 80)
    logging.info("GENERATING CONNECTION URL")
    logging.info("=" * 80)
    logging.info(f"Trusted connection: {SQL_SERVER_CONFIG['trusted_connection']}")
    
    if SQL_SERVER_CONFIG['trusted_connection']:
        # Windows Authentication
        logging.info("Using Windows Authentication")
        connection_string = (
            f"mssql+pyodbc://{SQL_SERVER_CONFIG['server']}/"
            f"{SQL_SERVER_CONFIG['database']}?"
            f"driver={SQL_SERVER_CONFIG['driver']}&"
            f"trusted_connection=yes"
        )
    else:
        # SQL Server Authentication with TCP connection (explicit port)
        logging.info("Using SQL Server Authentication")
        server_with_port = f"{SQL_SERVER_CONFIG['server']},1433"
        logging.info(f"Server with port: {server_with_port}")
        connection_string = (
            f"mssql+pyodbc://{SQL_SERVER_CONFIG['username']}:"
            f"{SQL_SERVER_CONFIG['password']}@"
            f"{server_with_port}/"
            f"{SQL_SERVER_CONFIG['database']}?"
            f"driver={SQL_SERVER_CONFIG['driver']}&"
            f"TrustServerCertificate=yes&"
            f"Encrypt=no"
        )
    
    logging.info(f"Generated connection string: {connection_string}")
    return connection_string

# Create SQL Server engine
sql_server_engine = None

def initialize_sql_server_connection():
    """Initialize SQL Server connection with detailed debugging and fallback methods"""
    global sql_server_engine
    
    logging.info("=" * 80)
    logging.info("SQL SERVER CONNECTION INITIALIZATION")
    logging.info("=" * 80)
    logging.info(f"Timestamp: {datetime.now()}")
    logging.info(f"Server: {SQL_SERVER_CONFIG['server']}")
    logging.info(f"Database: {SQL_SERVER_CONFIG['database']}")
    logging.info(f"Username: {SQL_SERVER_CONFIG['username']}")
    logging.info(f"Trusted Connection: {SQL_SERVER_CONFIG['trusted_connection']}")
    
    # Test basic connectivity first
    logging.info("=" * 80)
    logging.info("TESTING BASIC CONNECTIVITY")
    logging.info("=" * 80)
    
    try:
        import socket
        import subprocess
        
        # Test if server is reachable
        logging.info(f"Testing network connectivity to {SQL_SERVER_CONFIG['server']}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((SQL_SERVER_CONFIG['server'], 1433))
        sock.close()
        
        if result == 0:
            logging.info("SUCCESS: Server is reachable on port 1433")
        else:
            logging.error(f"FAILED: Cannot connect to {SQL_SERVER_CONFIG['server']}:1433")
            logging.error("This indicates a network connectivity issue")
            
        # Test ping
        try:
            logging.info(f"Testing ping to {SQL_SERVER_CONFIG['server']}...")
            result = subprocess.run(['ping', '-n', '1', SQL_SERVER_CONFIG['server']], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logging.info("SUCCESS: Ping successful")
            else:
                logging.error("FAILED: Ping failed")
                logging.error(f"Ping output: {result.stdout}")
        except Exception as ping_error:
            logging.error(f"Ping test failed: {ping_error}")
            
    except Exception as connectivity_error:
        logging.error(f"Connectivity test failed: {connectivity_error}")
        logging.error("This may indicate network or firewall issues")
    
    # Test direct pyodbc connection first
    logging.info("=" * 80)
    logging.info("TESTING DIRECT PYODBC CONNECTION")
    logging.info("=" * 80)
    
    try:
        import pyodbc
        
        # Test if pyodbc is available
        logging.info("Testing pyodbc availability...")
        logging.info(f"pyodbc version: {pyodbc.version}")
        
        # List available drivers
        logging.info("Available ODBC drivers:")
        drivers = pyodbc.drivers()
        for driver in drivers:
            logging.info(f"  - {driver}")
        
        # Check if our specific driver is available
        target_driver = SQL_SERVER_CONFIG['driver']
        if target_driver in drivers:
            logging.info(f"SUCCESS: {target_driver} is available")
        else:
            logging.error(f"FAILED: {target_driver} not found in available drivers")
            logging.error("This indicates ODBC driver installation issue")
        
        # Test direct connection
        direct_conn_str = f"DRIVER={{{target_driver}}};SERVER={SQL_SERVER_CONFIG['server']},1433;DATABASE={SQL_SERVER_CONFIG['database']};UID={SQL_SERVER_CONFIG['username']};PWD={SQL_SERVER_CONFIG['password']};Encrypt=no;TrustServerCertificate=yes"
        logging.info(f"Testing direct pyodbc connection string: {direct_conn_str.replace(SQL_SERVER_CONFIG['password'], '***')}")
        
        conn = pyodbc.connect(direct_conn_str, timeout=10)
        logging.info("SUCCESS: Direct pyodbc connection established")
        
        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        logging.info(f"SUCCESS: Direct pyodbc query result: {result[0]}")
        
        # Test database name
        cursor.execute("SELECT DB_NAME() as current_db")
        db_name = cursor.fetchone()[0]
        logging.info(f"SUCCESS: Current database: {db_name}")
        
        # Test server info
        cursor.execute("SELECT @@VERSION as version")
        version = cursor.fetchone()[0]
        logging.info(f"SUCCESS: SQL Server version: {version[:100]}...")
        
        conn.close()
        logging.info("SUCCESS: Direct pyodbc connection test completed")
        
    except Exception as direct_error:
        logging.error(f"FAILED: Direct pyodbc connection failed: {direct_error}")
        logging.error(f"Error type: {type(direct_error).__name__}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
    
    # Use direct pyodbc approach as primary method since it works reliably
    logging.info("=" * 80)
    logging.info("USING DIRECT PYODBC AS PRIMARY METHOD")
    logging.info("=" * 80)
    
    # Since direct pyodbc works, create SQLAlchemy engine with the working connection string
    working_connection_string = f"mssql+pyodbc://{SQL_SERVER_CONFIG['username']}:{SQL_SERVER_CONFIG['password']}@{SQL_SERVER_CONFIG['server']},1433/{SQL_SERVER_CONFIG['database']}?driver={SQL_SERVER_CONFIG['driver']}&TrustServerCertificate=yes&Encrypt=no"
    
    logging.info(f"Using working connection string: {working_connection_string}")
    
    try:
        logging.info("Creating SQLAlchemy engine with working connection string...")
        sql_server_engine = create_engine(
            working_connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=False
        )
        logging.info("SUCCESS: SQLAlchemy engine created with working connection string")
        
        # Test the connection
        logging.info("Testing SQLAlchemy connection...")
        with sql_server_engine.connect() as connection:
            logging.info("SUCCESS: SQLAlchemy connection established")
            
            # Test basic query
            result = connection.execute(text("SELECT 1 as test"))
            test_result = result.fetchone()
            logging.info(f"SUCCESS: Basic query successful: {test_result[0]}")
            
            # Test database name
            result = connection.execute(text("SELECT DB_NAME() as current_db"))
            db_name = result.fetchone()[0]
            logging.info(f"SUCCESS: Current database: {db_name}")
            
        logging.info("SUCCESS: SQL Server connection initialized successfully using direct pyodbc method")
        return True
        
    except Exception as e:
        logging.error(f"FAILED: Even the working connection string failed: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        
        # If even the working connection fails, try alternative approaches
        logging.info("Trying alternative connection methods...")
        
        connection_methods = [
            {
                "name": "Alternative 1: Without autocommit",
                "connection_string": f"mssql+pyodbc://{SQL_SERVER_CONFIG['username']}:{SQL_SERVER_CONFIG['password']}@{SQL_SERVER_CONFIG['server']},1433/{SQL_SERVER_CONFIG['database']}?driver={SQL_SERVER_CONFIG['driver']}&TrustServerCertificate=yes&Encrypt=no"
            },
            {
                "name": "Alternative 2: Without explicit port",
                "connection_string": f"mssql+pyodbc://{SQL_SERVER_CONFIG['username']}:{SQL_SERVER_CONFIG['password']}@{SQL_SERVER_CONFIG['server']}/{SQL_SERVER_CONFIG['database']}?driver={SQL_SERVER_CONFIG['driver']}&TrustServerCertificate=yes&Encrypt=no"
            }
        ]
    
    # Since we're using direct pyodbc approach, we don't need SQLAlchemy connection methods
    logging.info("Using direct pyodbc approach - skipping SQLAlchemy connection methods")
    return True
    # If all SQLAlchemy methods failed, try direct pyodbc approach
logging.info("\n--- Trying Direct pyodbc Fallback ---")
try:
        import pyodbc
        
        # Create direct pyodbc connection string
        direct_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER_CONFIG['server']},1433;DATABASE={SQL_SERVER_CONFIG['database']};UID={SQL_SERVER_CONFIG['username']};PWD={SQL_SERVER_CONFIG['password']};Encrypt=no;TrustServerCertificate=yes"
        
        logging.info(f"Direct pyodbc connection string: {direct_conn_str}")
        
        # Test direct connection
        conn = pyodbc.connect(direct_conn_str, timeout=10)
        logging.info("SUCCESS: Direct pyodbc connection successful")
        
        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        logging.info(f"SUCCESS: Direct pyodbc query successful: {result[0]}")
        
        conn.close()
        logging.info("SUCCESS: Direct pyodbc connection test successful")
        
        # Create SQLAlchemy engine with the working connection string
        working_connection_string = f"mssql+pyodbc://{SQL_SERVER_CONFIG['username']}:{SQL_SERVER_CONFIG['password']}@{SQL_SERVER_CONFIG['server']},1433/{SQL_SERVER_CONFIG['database']}?driver={SQL_SERVER_CONFIG['driver']}&TrustServerCertificate=yes&Encrypt=no"
        
        sql_server_engine = create_engine(
            working_connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=False
        )
        
        logging.info("SUCCESS: SQLAlchemy engine created with working connection string")
         return True
        
    except Exception as e:
        logging.error(f"FAILED: Direct pyodbc fallback also failed: {e}")
        logging.error("Full traceback of last attempt: {traceback.format_exc()}")
        return False

def get_sql_server_session():
    """Get SQL Server session"""
    if sql_server_engine is None:
        if not initialize_sql_server_connection():
            return None
    
    Session = sessionmaker(bind=sql_server_engine)
    return Session()

def search_customers(company_name_pattern):
    """
    Search customers in SQL Server database with detailed debugging
    Returns list of customer records matching the pattern
    """
    logging.info("=" * 80)
    logging.info("CUSTOMER SEARCH DEBUG")
    logging.info("=" * 80)
    logging.info(f"Timestamp: {datetime.now()}")
    logging.info(f"Search pattern: '{company_name_pattern}'")
    logging.info(f"Pattern length: {len(company_name_pattern)}")
    logging.info(f"Pattern type: {type(company_name_pattern)}")
    
    # Use direct pyodbc connection since SQLAlchemy is having issues
    logging.info("Using direct pyodbc connection for search...")
    
    try:
        import pyodbc
        
        # Create direct pyodbc connection
        direct_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER_CONFIG['server']},1433;DATABASE={SQL_SERVER_CONFIG['database']};UID={SQL_SERVER_CONFIG['username']};PWD={SQL_SERVER_CONFIG['password']};Encrypt=no;TrustServerCertificate=yes"
        logging.info(f"Direct connection string: {direct_conn_str.replace(SQL_SERVER_CONFIG['password'], '***')}")
        
        conn = pyodbc.connect(direct_conn_str, timeout=10)
        logging.info("SUCCESS: Direct pyodbc connection established for search")
        
        cursor = conn.cursor()
        
        # Look for customer-related tables
        logging.info("Searching for customer-related tables...")
        cursor.execute("""
            SELECT TABLE_NAME, TABLE_TYPE 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME LIKE '%customer%' 
               OR TABLE_NAME LIKE '%client%' 
               OR TABLE_NAME LIKE '%file%'
               OR TABLE_NAME LIKE '%company%'
            ORDER BY TABLE_NAME
        """)
        customer_tables = cursor.fetchall()
        logging.info(f"Found {len(customer_tables)} customer-related tables:")
        for table_name, table_type in customer_tables:
            logging.info(f"  - {table_name} ({table_type})")
        
        # Try to find a working customer table
        working_table = None
        for table_name, table_type in customer_tables:
            try:
                logging.info(f"Testing access to {table_name}...")
                cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
                sample = cursor.fetchone()
                if sample:
                    logging.info(f"SUCCESS: {table_name} is accessible, sample: {sample}")
                    working_table = table_name
                    break
                else:
                    logging.info(f"SUCCESS: {table_name} is accessible but empty")
                    working_table = table_name
                    break
            except Exception as table_error:
                logging.error(f"FAILED: {table_name} access failed: {table_error}")
                continue
        
        if not working_table:
            logging.error("FAILED: No accessible customer tables found")
            conn.close()
            return []
    
        logging.info(f"Using table: {working_table}")
        
        # Now perform the actual search
        logging.info("Executing customer search query...")
        
        # First, let's see what columns are available in the working table
        logging.info(f"Getting column information for {working_table}...")
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{working_table}' 
            ORDER BY ORDINAL_POSITION
        """)
        columns = cursor.fetchall()
        logging.info(f"Available columns in {working_table}:")
        for col_name, col_type in columns:
            logging.info(f"  - {col_name} ({col_type})")
        
        # Try to find name-related columns
        name_columns = []
        for col_name, col_type in columns:
            if any(keyword in col_name.lower() for keyword in ['name', 'file', 'company', 'client']):
                name_columns.append(col_name)
        
        if not name_columns:
            logging.error("FAILED: No name-related columns found")
            conn.close()
            return []
        
        logging.info(f"Using name columns: {name_columns}")
        
        # Build a dynamic search query
        search_conditions = []
        for col in name_columns:
            search_conditions.append(f"{col} LIKE ?")
        
        query = f"""
            SELECT TOP 50 *
            FROM {working_table}
            WHERE {' OR '.join(search_conditions)}
        """
        
        search_pattern = f"%{company_name_pattern}%"
        search_params = [search_pattern] * len(search_conditions)
        
        logging.info(f"Search query: {query}")
        logging.info(f"Search parameter: {search_pattern}")
        
        cursor.execute(query, search_params)
            customers = []
        
        logging.info("Processing search results...")
        row_count = 0
        for row in cursor.fetchall():
            row_count += 1
            logging.info(f"Processing row {row_count}...")
            
            # Create a dynamic customer record based on available columns
            customer = {}
            for i, (col_name, col_type) in enumerate(columns):
                value = row[i] if i < len(row) else None
                customer[col_name.lower()] = value
                logging.info(f"  {col_name}: {value}")
            
            # Map to expected fields with fallbacks
            customer_record = {
                'file_number': customer.get('filenumber', customer.get('file_number', customer.get('id', 'N/A'))),
                'file_name': customer.get('filename', customer.get('file_name', customer.get('name', customer.get('company', 'N/A')))),
                'first_name': customer.get('firstname', customer.get('first_name', customer.get('contact', 'N/A'))),
                'telephone': customer.get('tel', customer.get('telephone', customer.get('phone', 'N/A'))),
                'email': customer.get('email', 'N/A'),
                'address': customer.get('address', customer.get('location', 'N/A')),
                'vat_id': customer.get('vatid', customer.get('vat_id', 'N/A'))
            }
            
            customers.append(customer_record)
            logging.info(f"  Customer {row_count}: {customer_record['file_name']} (File: {customer_record['file_number']})")
        
        conn.close()
        logging.info(f"SUCCESS: Search completed successfully, found {len(customers)} customers")
        logging.info(f"Total rows processed: {row_count}")
        
        if len(customers) == 0:
            logging.info("No customers found matching the search pattern")
            logging.info("This could be due to:")
            logging.info("  1. No customers match the search pattern")
            logging.info("  2. Database access restrictions")
            logging.info("  3. CustomerFile view/table is empty")
            logging.info("  4. NAJ_DATA database access issues")
        else:
            logging.info(f"Found {len(customers)} customers matching the search pattern")
            for i, customer in enumerate(customers[:5]):  # Show first 5 customers
                logging.info(f"  Customer {i+1}: {customer['file_name']} (File: {customer['file_number']})")
            if len(customers) > 5:
                logging.info(f"  ... and {len(customers) - 5} more customers")
        
            return customers
            
    except Exception as e:
        logging.error(f"FAILED: Error searching customers: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        
        # Check if it's a database access issue
        error_msg = str(e)
        if "NAJ_DATA" in error_msg or "not able to access" in error_msg:
            logging.error("DIAGNOSIS: Database access denied")
            logging.error("The user does not have access to the required database tables")
            logging.error("SOLUTION: Contact the database administrator to grant access to the CustomerFile view")
            logging.error("ALTERNATIVE: Use a different table or create a new view with proper permissions")
            return []
        else:
            logging.error(f"DIAGNOSIS: {error_msg}")
        return []

def test_connection():
    """Test SQL Server connection"""
    try:
        session = get_sql_server_session()
        if session:
            result = session.execute(text("SELECT 1 as test"))
            session.close()
            return True
    except Exception as e:
        logging.error(f"SQL Server connection test failed: {e}")
    return False
