import sqlite3
import os
import json

def inspect_database(db_path):
    """
    Inspect a SQLite database file and print information about its structure and contents.
    
    Args:
        db_path: Path to the SQLite database file
    """
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return
    
    print(f"Inspecting database: {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            conn.close()
            return
        
        print(f"\nFound {len(tables)} tables:")
        
        # Iterate through each table
        for table in tables:
            table_name = table[0]
            print(f"\n{'=' * 50}")
            print(f"Table: {table_name}")
            print(f"{'=' * 50}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            print("\nSchema:")
            for col in columns:
                col_id, name, dtype, notnull, default_val, pk = col
                print(f"  {name} ({dtype})", end="")
                if pk:
                    print(" PRIMARY KEY", end="")
                if notnull:
                    print(" NOT NULL", end="")
                if default_val is not None:
                    print(f" DEFAULT {default_val}", end="")
                print()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"\nRow count: {row_count}")
            
            # Show sample data (up to 5 rows)
            if row_count > 0:
                print("\nSample data:")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                rows = cursor.fetchall()
                
                # Display column names
                col_names = [col[1] for col in columns]
                print("  " + " | ".join(col_names))
                print("  " + "-" * 50)
                
                # Display data
                for row in rows:
                    # Format data for display
                    formatted_row = []
                    for item in row:
                        if isinstance(item, str) and len(item) > 20:
                            formatted_row.append(f"{item[:20]}...")
                        else:
                            formatted_row.append(str(item))
                    print("  " + " | ".join(formatted_row))
                
                if row_count > 5:
                    print(f"  ... {row_count - 5} more rows")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    db_path = "database/baijiu.db"
    inspect_database(db_path) 