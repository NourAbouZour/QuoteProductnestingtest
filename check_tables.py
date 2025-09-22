#!/usr/bin/env python3
"""Check what tables exist in the database"""

from DatabaseConfig import engine
from sqlalchemy import text

try:
    conn = engine.connect()
    # Use SQLAlchemy connection, not psycopg2 style
    
    # Get all tables
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"))
    tables = result.fetchall()
    
    print('Available tables:')
    for table in tables:
        print(f'  - {table[0]}')
    
    # Check if boards table exists and get its structure
    table_names = [table[0] for table in tables]
    if 'boards' in table_names:
        print('\nBoards table structure:')
        result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'boards' ORDER BY ordinal_position;"))
        columns = result.fetchall()
        for col in columns:
            print(f'  - {col[0]}: {col[1]}')
        
        # Get sample data
        print('\nSample boards data:')
        result = conn.execute(text("SELECT * FROM boards LIMIT 5;"))
        rows = result.fetchall()
        for row in rows:
            print(f'  {row}')
    else:
        print('\n❌ No "boards" table found')
    
    conn.close()
    
except Exception as e:
    print(f'Error: {e}')
