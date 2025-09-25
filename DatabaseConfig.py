"""
Database Configuration for DXF Analyzer
PostgreSQL connection and session management
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import event
import logging

# Database configuration
DATABASE_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'QuotationDB'),
    'user': os.environ.get('DB_USER', 'dxf_user'),
    'password': os.environ.get('DB_PASSWORD', 'DXFanalyzer2024!'),
}

# Construct database URL; allow overriding schema via env var DB_SCHEMA
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
DB_SCHEMA = os.environ.get('DB_SCHEMA')  # optional

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    echo=False           # Set to True for SQL logging
)

# Optionally set search_path for a specific schema if provided
if DB_SCHEMA:
    @event.listens_for(engine, "connect")
    def set_search_path_on_connect(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET search_path TO {DB_SCHEMA}, public")
            cursor.close()
        except Exception as e:
            # Non-fatal; will default to public
            import sys
            print(f"[db] warning: failed to set search_path on connect: {e}", file=sys.stderr)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Get database session
    Use this function to get a database session for your routes
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

_SCRAP_COL_CACHED = None

def _get_materials_columns(conn):
    try:
        rows = conn.execute(text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'materials'
            """
        )).fetchall()
        return {str(r[0]) for r in rows}
    except Exception:
        return set()

def _resolve_scrap_column_name(conn):
    global _SCRAP_COL_CACHED
    if _SCRAP_COL_CACHED:
        return _SCRAP_COL_CACHED
    cols = _get_materials_columns(conn)
    # Normalize candidates by case for matching
    lower_to_original = {c.lower(): c for c in cols}
    candidates = [
        'scrap price per kg',
        'scrap price',
        'scrap_price_per_kg',
        'scrappriceperkg',
        'scrap price/kg'
    ]
    for cand in candidates:
        if cand in lower_to_original:
            _SCRAP_COL_CACHED = lower_to_original[cand]
            return _SCRAP_COL_CACHED
    # Default to standardized name if present
    if 'Scrap Price per kg' in cols:
        _SCRAP_COL_CACHED = 'Scrap Price per kg'
        return _SCRAP_COL_CACHED
    return None

def _scrap_select_alias(conn):
    """Return a SQL fragment selecting scrap price aliased as "Scrap Price per kg"."""
    name = _resolve_scrap_column_name(conn)
    if name:
        # Quote identifier with double quotes, escape any embedded quotes
        q = '"' + name.replace('"', '""') + '"'
        return f"COALESCE({q}, 0) AS \"Scrap Price per kg\""
    else:
        return '0 AS "Scrap Price per kg"'

def ensure_materials_schema():
    """Ensure materials table has required columns."""
    try:
        with engine.connect() as conn:
            cols = _get_materials_columns(conn)
            if 'Scrap Price per kg' not in cols:
                try:
                    conn.execute(text('ALTER TABLE materials ADD COLUMN "Scrap Price per kg" DOUBLE PRECISION'))
                    conn.commit()
                except Exception:
                    conn.rollback()
            # Backfill from legacy columns where available
            legacy_cols = [
                'Scrap Price per kg',  # already standard
                'Scrap Price',
                'scrap_price_per_kg',
                'Scrap price per kg',
                'Scrap price',
                'Scrap Price/kg'
            ]
            for lc in legacy_cols:
                # Skip if legacy col does not exist
                if lc not in _get_materials_columns(conn):
                    continue
                try:
                    q_legacy = '"' + lc.replace('"', '""') + '"'
                    conn.execute(text(
                        f'UPDATE materials SET "Scrap Price per kg" = {q_legacy} '
                        f'WHERE ("Scrap Price per kg" IS NULL OR "Scrap Price per kg" = 0) AND {q_legacy} IS NOT NULL'
                    ))
                    conn.commit()
                except Exception:
                    conn.rollback()
    except Exception:
        # Non-fatal; app will continue and queries using this column are guarded with COALESCE
        pass

# Attempt to ensure schema on import
ensure_materials_schema()

def get_db_direct():
    """
    Get database session directly (for non-generator contexts)
    Remember to close the session when done!
    """
    return SessionLocal()

def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return False

def execute_raw_sql(sql, params=None):
    """Execute raw SQL query"""
    try:
        with engine.connect() as connection:
            if params:
                result = connection.execute(text(sql), params)
            else:
                result = connection.execute(text(sql))
            connection.commit()
            return result
    except Exception as e:
        logging.error(f"SQL execution failed: {e}")
        raise

def get_table_row_count(table_name):
    """Get row count for a specific table"""
    try:
        result = execute_raw_sql(f"SELECT COUNT(*) FROM {table_name}")
        return result.fetchone()[0]
    except Exception as e:
        logging.error(f"Failed to get row count for {table_name}: {e}")
        return 0

# Database health check function
def check_database_health():
    """Perform comprehensive database health check"""
    health_status = {
        'connection': False,
        'tables': {},
        'total_records': 0
    }
    
    try:
        # Test connection
        health_status['connection'] = test_connection()
        
        if health_status['connection']:
            # Check tables
            tables = [
                 'materials'
            ]
            
            for table in tables:
                try:
                    count = get_table_row_count(table)
                    health_status['tables'][table] = count
                    health_status['total_records'] += count
                except Exception as e:
                    health_status['tables'][table] = f"Error: {e}"
        
        return health_status
        
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        health_status['error'] = str(e)
        return health_status

def get_materials_data():
    """Get all materials data for dropdowns"""
    try:
        with engine.connect() as conn:
            cursor = conn.execute(text("""
                SELECT DISTINCT "Material Name" 
                FROM materials 
                WHERE "Material Name" IS NOT NULL 
                ORDER BY "Material Name"
            """))
            material_names = [row[0] for row in cursor.fetchall()]
            
            cursor = conn.execute(text("""
                SELECT DISTINCT "Thickness" 
                FROM materials 
                WHERE "Thickness" IS NOT NULL 
                ORDER BY "Thickness"
            """))
            thicknesses = [float(row[0]) for row in cursor.fetchall()]
            
            cursor = conn.execute(text("""
                SELECT DISTINCT "Grade" 
                FROM materials 
                WHERE "Grade" IS NOT NULL 
                ORDER BY "Grade"
            """))
            grades = [row[0] for row in cursor.fetchall()]
            
            cursor = conn.execute(text("""
                SELECT DISTINCT "Finish" 
                FROM materials 
                WHERE "Finish" IS NOT NULL 
                ORDER BY "Finish"
            """))
            finishes = [row[0] for row in cursor.fetchall()]
            
            return {
                'material_names': material_names,
                'material_types': material_names,  # Add alias for frontend compatibility
                'thicknesses': thicknesses,
                'grades': grades,
                'finishes': finishes
            }
        

    except Exception as e:
        print(f"Error getting materials data: {e}")
        return {
            'material_types': [],
            'thicknesses': [],
            'grades': [],
            'finishes': []
        }

def get_filtered_options(material_name=None, thickness=None, grade=None, finish=None):
    """Get filtered options based on selected values"""
    try:
        with engine.connect() as conn:
            # Build WHERE clause based on provided filters
            where_conditions = []
            params = {}
            
            if material_name:
                where_conditions.append('"Material Name" = :material_name')
                params['material_name'] = material_name
            if thickness is not None:
                where_conditions.append('"Thickness" = :thickness')
                params['thickness'] = thickness
            if grade is not None:
                # Handle grade properly - 0 is a valid grade for materials like Brass
                try:
                    params['grade'] = int(grade)
                except Exception:
                    params['grade'] = 0
                where_conditions.append('"Grade" = :grade')
            if finish is not None:
                # Handle finish properly - 0 is a valid finish for materials like Brass
                params['finish'] = str(finish)
                where_conditions.append('"Finish" = :finish')
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Get material types (only when no material filter is applied)
            material_types = []
            if not material_name:
                cursor = conn.execute(text("""
                    SELECT DISTINCT "Material Name" 
                    FROM materials 
                    WHERE "Material Name" IS NOT NULL 
                    ORDER BY "Material Name"
                """))
                material_types = [row[0] for row in cursor.fetchall()]
            
            # Get filtered thicknesses
            cursor = conn.execute(text(f"""
                SELECT DISTINCT "Thickness" 
                FROM materials 
                WHERE {where_clause} AND "Thickness" IS NOT NULL 
                ORDER BY "Thickness"
            """), params)
            thicknesses = [float(row[0]) for row in cursor.fetchall()]
            
            # Get filtered grades
            cursor = conn.execute(text(f"""
                SELECT DISTINCT "Grade" 
                FROM materials 
                WHERE {where_clause} AND "Grade" IS NOT NULL 
                ORDER BY "Grade"
            """), params)
            grades = [row[0] for row in cursor.fetchall()]
            
            # Get filtered finishes
            cursor = conn.execute(text(f"""
                SELECT DISTINCT "Finish" 
                FROM materials 
                WHERE {where_clause} AND "Finish" IS NOT NULL 
                ORDER BY "Finish"
            """), params)
            finishes = [row[0] for row in cursor.fetchall()]
            
            return {
                'material_types': material_types,
                'thicknesses': thicknesses,
                'grades': grades,
                'finishes': finishes
            }
    except Exception as e:
        print(f"Error getting filtered options: {e}")
        return {
            'material_types': [],
            'thicknesses': [],
            'grades': [],
            'finishes': []
        }

def get_standard_boards():
    """
    Fetch standard boards from database, sorted by size (largest first).
    Expected schema for table 'boards':
      id SERIAL/INT PK, length_mm FLOAT, width_mm FLOAT, quantity INT
    Returns list of dicts: [{id, length_mm, width_mm, quantity, area_sq_mm}, ...]
    Sorted by area (largest first) for optimal nesting algorithm performance
    """
    try:
        with engine.connect() as conn:
            # Detect if 'boards' table exists in the current database
            try:
                # Works for PostgreSQL; adjust for other DBs if needed
                exists = conn.execute(text("""
                    SELECT to_regclass('public.boards') IS NOT NULL
                """)).scalar()
                if not exists:
                    # Fallback generic detection
                    try:
                        _ = conn.execute(text("SELECT 1 FROM boards LIMIT 1"))
                        exists = True
                    except Exception:
                        exists = False
            except Exception:
                # Fallback generic detection
                try:
                    _ = conn.execute(text("SELECT 1 FROM boards LIMIT 1"))
                    exists = True
                except Exception:
                    exists = False

            if not exists:
                print("[boards] Table 'boards' not found; returning empty list.")
                return []

            # Enhanced query to calculate area and sort by size (largest first)
            # Note: Based on the image, the columns are 'length' and 'width', not 'length_mm' and 'width_mm'
            rows = conn.execute(text('''
                SELECT 
                    id, 
                    length, 
                    width, 
                    COALESCE(quantity, 0) AS quantity,
                    (length * width) AS area_sq_mm
                FROM boards 
                WHERE length > 0 AND width > 0
                ORDER BY area_sq_mm DESC, length DESC, width DESC
            ''')).fetchall()
            
            boards = []
            for r in rows:
                try:
                    board_data = {
                        'id': int(r[0]),
                        'length_mm': float(r[1]),  # Convert length to mm
                        'width_mm': float(r[2]),   # Convert width to mm
                        'quantity': int(r[3]),
                        'area_sq_mm': float(r[4])
                    }
                    boards.append(board_data)
                    print(f"[boards] Loaded board ID {board_data['id']}: {board_data['length_mm']:.1f}x{board_data['width_mm']:.1f}mm (area: {board_data['area_sq_mm']:.0f} sq mm) qty: {board_data['quantity']}")
                except Exception as e:
                    print(f"[boards] Error processing board row: {e}")
                    continue
            
            print(f"[boards] Total boards loaded: {len(boards)} (sorted by size, largest first)")
            return boards
    except Exception as e:
        print(f"[boards] Error fetching boards: {e}")
        return []

def get_material_config(material_name, thickness, grade, finish):
    """Get material configuration for cost calculation"""
    try:
        # Debug logging
        print(f"\n=== get_material_config DEBUG ===")
        print(f"Input params: material_name='{material_name}' (type: {type(material_name)})")
        print(f"Input params: thickness='{thickness}' (type: {type(thickness)})")
        print(f"Input params: grade='{grade}' (type: {type(grade)})")
        print(f"Input params: finish='{finish}' (type: {type(finish)})")
        
        with engine.connect() as conn:
            # Convert parameters to proper types for exact matching
            # Handle None, empty string, and 0 values properly
            if grade is None or grade == '':
                exact_grade = 0
            else:
                try:
                    exact_grade = int(grade)
                except (ValueError, TypeError):
                    exact_grade = 0
            
            if finish is None or finish == '':
                exact_finish = '0'
            else:
                exact_finish = str(finish)
            
            # Query for the exact material row based on all four parameters
            # Be resilient to type inconsistencies in the DB (e.g., Finish stored as text '0' or numeric 0)
            scrap_alias = _scrap_select_alias(conn)
            exact_match_sql = f"""
                SELECT "Speed", "Vaporization Speed", "Piercing Time", "Price per kg", 
                       "Density", "V-Groov", "Bending", {scrap_alias}
                FROM materials 
                WHERE LOWER(TRIM("Material Name")) = LOWER(TRIM(:material_name))
                  AND "Thickness" = :thickness
                  AND COALESCE(CAST("Grade" AS INTEGER), 0) = :grade
                  AND COALESCE(TRIM(CAST("Finish" AS TEXT)), '0') = TRIM(:finish)
                LIMIT 1
            """
            
            exact_params = {
                'material_name': material_name,
                'thickness': thickness,
                'grade': exact_grade,
                'finish': exact_finish
            }
            
            print(f"  🔍 Trying exact match with params: {exact_params}")
            result = None
            try:
                cursor = conn.execute(text(exact_match_sql), exact_params)
                result = cursor.fetchone()
            except Exception as e:
                # If CAST fails for any reason, fall back to a simpler exact match
                print(f"  ⚠ Exact match with CAST failed: {e}. Falling back to simpler exact match.")
                fallback_exact_sql = f"""
                    SELECT "Speed", "Vaporization Speed", "Piercing Time", "Price per kg", 
                           "Density", "V-Groov", "Bending", {scrap_alias}
                    FROM materials 
                    WHERE LOWER(TRIM("Material Name")) = LOWER(TRIM(:material_name))
                      AND "Thickness" = :thickness
                      AND COALESCE("Grade", 0) = :grade
                      AND COALESCE(TRIM("Finish"::text), '0') = TRIM(:finish)
                    LIMIT 1
                """
                cursor = conn.execute(text(fallback_exact_sql), exact_params)
                result = cursor.fetchone()
            
            if result:
                print(f"  ✓ Exact match found in database")
                return {
                    'machine_speed': float(result[0]) if result[0] is not None else 100.0,
                    'vaporization_speed': float(result[1]) if result[1] is not None else 50.0,
                    'piercing_time': float(result[2]) if result[2] is not None else 0.5,
                    'price_per_kg': float(result[3]) if result[3] is not None else 25.0,
                    'density': float(result[4]) if result[4] is not None else 7.85,
                    'vgroove_price': float(result[5]) if result[5] is not None else 0.0,
                    'bending_price': float(result[6]) if result[6] is not None else 0.0,
                    'scrap_price_per_kg': float(result[7]) if len(result) > 7 and result[7] is not None else 0.0,
                    'found': True
                }
            
            # If exact match not found, try progressive fallbacks
            print(f"  ❌ Exact match not found for: {material_name} / {thickness} / {exact_grade} / {exact_finish}")
            
            # Fallback 1: Match by material and thickness only (ignore grade/finish)
            try:
                loose_sql = f"""
                    SELECT "Speed", "Vaporization Speed", "Piercing Time", "Price per kg", 
                           "Density", "V-Groov", "Bending", {scrap_alias}
                    FROM materials 
                    WHERE LOWER(TRIM("Material Name")) = LOWER(TRIM(:material_name))
                      AND "Thickness" = :thickness
                    ORDER BY COALESCE(CAST("Grade" AS INTEGER), 0) ASC, COALESCE(CAST("Finish" AS TEXT), '0') ASC
                    LIMIT 1
                """
                loose_params = {
                    'material_name': material_name,
                    'thickness': thickness
                }
                cursor = conn.execute(text(loose_sql), loose_params)
                loose_result = cursor.fetchone()
                if loose_result:
                    print("  ✓ Fallback match found by material + thickness only")
                    return {
                        'machine_speed': float(loose_result[0]) if loose_result[0] is not None else 100.0,
                        'vaporization_speed': float(loose_result[1]) if loose_result[1] is not None else 50.0,
                        'piercing_time': float(loose_result[2]) if loose_result[2] is not None else 0.5,
                        'price_per_kg': float(loose_result[3]) if loose_result[3] is not None else 25.0,
                        'density': float(loose_result[4]) if loose_result[4] is not None else 7.85,
                        'vgroove_price': float(loose_result[5]) if loose_result[5] is not None else 0.0,
                            'bending_price': float(loose_result[6]) if loose_result[6] is not None else 0.0,
                            'scrap_price_per_kg': float(loose_result[7]) if len(loose_result) > 7 and loose_result[7] is not None else 0.0,
                        'found': True
                    }
            except Exception as e:
                print(f"  ⚠ Loose match by material+thickness failed: {e}")
            
            # Debug: Show what's available for this material to aid troubleshooting
            try:
                debug_result = conn.execute(text('SELECT TRIM("Material Name") AS name, "Thickness", "Grade", "Finish" FROM materials WHERE LOWER(TRIM("Material Name")) = LOWER(TRIM(:material_name)) ORDER BY "Thickness", COALESCE("Grade",0), COALESCE("Finish",\'0\')'), {'material_name': material_name}).fetchall()
                print(f"  🔍 Available entries for {material_name}: {debug_result}")
            except Exception as e:
                print(f"  ⚠ Debug listing failed: {e}")
            
            # Not found in DB
            print(f"  ❌ Material config NOT found in database")
            print(f"      Searched for: {material_name} / {thickness} / {exact_grade} / {exact_finish}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error getting material config: {e}")
        return None

def get_all_materials():
    """Get all materials for admin panel with unique identification"""
    try:
        with engine.connect() as conn:
            scrap_alias = _scrap_select_alias(conn)
            sql = f"""
                SELECT 
                    "Material Name",
                    COALESCE("Grade", 0) AS "Grade",
                    COALESCE("Finish", '0') AS "Finish",
                    "Thickness",
                    "Density",
                    "Price per kg",
                    {scrap_alias},
                    "Speed",
                    "Piercing Time",
                    COALESCE("Vaporization Speed", 0) AS "Vaporization Speed",
                    "V-Groov",
                    "Bending"
                FROM materials 
                ORDER BY "Material Name", "Thickness", COALESCE("Grade", 0), COALESCE("Finish", '0')
            """
            result = conn.execute(text(sql))
            
            # Get column names from the result
            columns = result.keys()
            materials = []
            
            for row in result.fetchall():
                material = dict(zip(columns, row))
                # Convert Decimal to float for JSON serialization
                for key, value in material.items():
                    if hasattr(value, '__float__'):
                        material[key] = float(value)
                
                # Create a unique identifier for each material combination
                material['unique_id'] = f"{material['Material Name']}_{material['Thickness']}_{material.get('Grade', '')}_{material.get('Finish', '')}"
                materials.append(material)
            
            return materials
    except Exception as e:
        print(f"Error getting all materials: {e}")
        return []

def update_material(original_material_name, data, original_thickness=None, original_grade=None, original_finish=None):
    """Update material in database - Update only the specific material row"""
    try:
        with engine.connect() as conn:
            print(f"DEBUG: Looking for material '{original_material_name}' to update")
            
            # If we don't have the original identifying info, we need to get it from the edit form data
            # The edit form should send the ORIGINAL thickness that identifies which specific material to update
            if original_thickness is None:
                # This is a problem - we need to know which specific material to update
                # For now, let's use the thickness from the form data as the identifier
                original_thickness = data['thickness']
                original_grade = data['grade']
                original_finish = data['finish']
                print(f"DEBUG: Using form data as identifier - Thickness: {original_thickness}, Grade: {original_grade}, Finish: '{original_finish}'")
            
            # Convert to proper types
            original_thickness = float(original_thickness)
            original_grade = int(original_grade) if original_grade is not None else 0
            original_finish = str(original_finish) if original_finish is not None else ''
            
            print(f"DEBUG: Looking for EXACT material - Name: '{original_material_name}', Thickness: {original_thickness}, Grade: {original_grade}, Finish: '{original_finish}'")
            
            # First, verify the material exists
            check_result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM materials 
                WHERE "Material Name" = :material_name 
                  AND "Thickness" = :thickness
                  AND COALESCE("Grade", 0) = :grade
                  AND COALESCE("Finish", '0') = :finish
            """), {
                'material_name': original_material_name,
                'thickness': original_thickness,
                'grade': original_grade,
                'finish': original_finish
            })
            
            count = check_result.fetchone()[0]
            print(f"DEBUG: Found {count} materials matching the criteria")
            
            if count == 0:
                print(f"DEBUG: No material found to update")
                return False
            elif count > 1:
                print(f"DEBUG: WARNING - Multiple materials found, this should not happen!")
                return False
            
            # Now update the specific material
            update_params = {
                'new_material_name': data['material_name'],
                'new_grade': data['grade'],
                'new_finish': data['finish'],
                'new_thickness': data['thickness'],
                'new_density': data['density'],
                'new_price_per_kg': data['price_per_kg'],
                'new_scrap_price_per_kg': data.get('scrap_price_per_kg', 0.0),
                'new_speed': data['speed'],
                'new_piercing_time': data['piercing_time'],
                'new_vaporization_speed': data['vaporization_speed'],
                'new_vgroove_price': data['vgroove_price'],
                'new_bending_price': data['bending_price'],
                # Original identifying fields
                'original_material_name': original_material_name,
                'original_thickness': original_thickness,
                'original_grade': original_grade,
                'original_finish': original_finish
            }
            
            # Update the exact material
            result = conn.execute(text("""
                UPDATE materials 
                SET "Material Name" = :new_material_name,
                    "Grade" = :new_grade, 
                    "Finish" = :new_finish,
                    "Thickness" = :new_thickness, 
                    "Density" = :new_density, 
                    "Price per kg" = :new_price_per_kg,
                    "Scrap Price per kg" = :new_scrap_price_per_kg,
                    "Speed" = :new_speed, 
                    "Piercing Time" = :new_piercing_time,
                    "Vaporization Speed" = :new_vaporization_speed, 
                    "V-Groov" = :new_vgroove_price,
                    "Bending" = :new_bending_price
                WHERE "Material Name" = :original_material_name 
                  AND "Thickness" = :original_thickness
                  AND COALESCE("Grade", 0) = :original_grade
                  AND COALESCE("Finish", '0') = :original_finish
            """), update_params)
            
            rows_affected = result.rowcount
            print(f"DEBUG: Update affected {rows_affected} rows")
            
            if rows_affected != 1:
                print(f"DEBUG: WARNING - Expected to update 1 row, but updated {rows_affected} rows!")
                return False
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"ERROR updating material: {e}")
        return False

def add_material(data):
    """Add new material to database"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO materials ("Material Name", "Grade", "Finish", "Thickness", 
                                     "Density", "Price per kg", "Scrap Price per kg", "Speed", "Piercing Time", 
                                     "Vaporization Speed", "V-Groov", "Bending")
                VALUES (:material_name, :grade, :finish, :thickness, :density, 
                        :price_per_kg, :scrap_price_per_kg, :speed, :piercing_time, :vaporization_speed, 
                        :vgroove_price, :bending_price)
            """), {
                'material_name': data['material_name'],
                'grade': data['grade'],
                'finish': data['finish'],
                'thickness': data['thickness'],
                'density': data['density'],
                'price_per_kg': data['price_per_kg'],
                'scrap_price_per_kg': data.get('scrap_price_per_kg', 0.0),
                'speed': data['speed'],
                'piercing_time': data['piercing_time'],
                'vaporization_speed': data['vaporization_speed'],
                'vgroove_price': data['vgroove_price'],
                'bending_price': data['bending_price']
            })
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error adding material: {e}")
        return False

def delete_material(material_name, thickness, grade=None, finish=None):
    """Delete material from database with unique identification"""
    try:
        with engine.connect() as conn:
            print(f"DEBUG: Deleting material - Name: '{material_name}', Thickness: {thickness}, Grade: {grade}, Finish: '{finish}'")
            
            # Convert to proper types for consistent matching
            thickness = float(thickness)
            grade = int(grade) if grade is not None and grade != '' else 0
            finish = str(finish) if finish is not None else '0'
            
            print(f"DEBUG: Converted values - Thickness: {thickness}, Grade: {grade}, Finish: '{finish}'")
            
            # First, check how many materials match the criteria
            check_result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM materials 
                WHERE "Material Name" = :material_name 
                  AND "Thickness" = :thickness
                  AND COALESCE("Grade", 0) = :grade
                  AND COALESCE("Finish", '0') = :finish
            """), {
                'material_name': material_name,
                'thickness': thickness,
                'grade': grade,
                'finish': finish
            })
            
            count = check_result.fetchone()[0]
            print(f"DEBUG: Found {count} materials matching the criteria for deletion")
            
            if count == 0:
                print(f"DEBUG: No material found to delete")
                return False
            elif count > 1:
                print(f"DEBUG: WARNING - Multiple materials found for deletion, this should not happen!")
                return False
            
            # Delete the specific material using COALESCE for consistent matching
            result = conn.execute(text("""
                DELETE FROM materials 
                WHERE "Material Name" = :material_name 
                  AND "Thickness" = :thickness
                  AND COALESCE("Grade", 0) = :grade
                  AND COALESCE("Finish", '0') = :finish
            """), {
                'material_name': material_name,
                'thickness': thickness,
                'grade': grade,
                'finish': finish
            })
            
            rows_affected = result.rowcount
            print(f"DEBUG: Delete affected {rows_affected} rows")
            
            if rows_affected != 1:
                print(f"DEBUG: WARNING - Expected to delete 1 row, but deleted {rows_affected} rows!")
                return False
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"ERROR deleting material: {e}")
        return False

def create_users_table():
    """Create users table if it doesn't exist"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    work_id VARCHAR(255),
                    extnumber INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error creating users table: {e}")
        return False

def get_user_by_email(email):
    """Get user by email"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, email, password, full_name, work_id, extnumber
                FROM users 
                WHERE email = :email
            """), {'email': email})
            
            row = result.fetchone()
            if row:
                user_data = {
                    'id': row[0],
                    'email': row[1],
                    'password': row[2],
                    'full_name': row[3],
                    'work_id': row[4],
                    'extnumber': row[5]
                }
                print(f"DEBUG: Database query result for {email}: {user_data}")
                return user_data
            return None
    except Exception as e:
        print(f"Error getting user by email: {e}")
        return None

def create_user(email, password, full_name, work_id=None, extnumber=None):
    """Create a new user"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO users (email, password, full_name, work_id, extnumber)
                VALUES (:email, :password, :full_name, :work_id, :extnumber)
            """), {
                'email': email,
                'password': password,
                'full_name': full_name,
                'work_id': work_id,
                'extnumber': extnumber
            })
            conn.commit()
            return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

if __name__ == "__main__":
    # Test database connection when run directly
    print("🔍 Testing PostgreSQL connection...")
    
    if test_connection():
        print("✅ Database connection successful!")
        
        # Create users table
        print("📋 Creating users table...")
        if create_users_table():
            print("✅ Users table created/verified!")
        else:
            print("❌ Failed to create users table!")
        
        print("\n📊 Database health check:")
        health = check_database_health()
        
        for table, count in health['tables'].items():
            print(f"   {table}: {count} records")
        
        print(f"\n📈 Total records: {health['total_records']}")
    else:
        print("❌ Database connection failed!")
        print("Please check your PostgreSQL setup and configuration.") 