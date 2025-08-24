# keyword_manager.py (Version 2.0 - More flexible)
import sys
import csv
import json
import os
from storage import XBotDetectorDB  # Correctly import the class from 'storage.py'

# Default configuration
DEFAULT_DATA_FILE = 'data/keywords.json'

def load_keywords_from_file(file_path):
    """Loads keywords from a CSV or JSON file based on its extension."""
    if not file_path:
        print("Error: No file path provided.")
        return []
    
    print(f"Attempting to load data from: {file_path}")
    
    if file_path.endswith('.csv'):
        try:
            with open(file_path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                keywords = [row for row in reader]
                print(f"-> Successfully loaded {len(keywords)} keywords from CSV.")
                return keywords
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
        except Exception as e:
            print(f"Error reading CSV: {e}")
        return []
    
    elif file_path.endswith('.json'):
        try:
            with open(file_path, mode='r', encoding='utf-8') as f:
                keywords = json.load(f)
                print(f"-> Successfully loaded {len(keywords)} keywords from JSON.")
                return keywords
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file. Please check for formatting errors. Details: {e}")
        return []
        
    else:
        print(f"Error: Unsupported file format for '{file_path}'. Please provide a .csv or .json file.")
        return []

def main():
    """Main function to parse the data file and load it into MongoDB."""
    
    # Use default file path if no command-line argument provided
    if len(sys.argv) >= 2:
        input_file_path = sys.argv[1]
        print(f"Using provided file path: {input_file_path}")
    else:
        input_file_path = DEFAULT_DATA_FILE
        print(f"No file path provided. Using default: {input_file_path}")
    
    # Check if file exists
    if not os.path.exists(input_file_path):
        print(f"\n❌ Error: File not found at '{input_file_path}'")
        if input_file_path == DEFAULT_DATA_FILE:
            print("   Please ensure the 'data' directory exists and contains 'keywords.json'")
        print("\n   Usage examples:")
        print("   python keyword_manager.py                    # Uses default data/keywords.json")
        print("   python keyword_manager.py data/my_file.csv   # Uses custom file")
        print("   python keyword_manager.py data/my_file.json  # Uses custom JSON file\n")
        return
    
    keywords_to_load = load_keywords_from_file(input_file_path)
    
    if not keywords_to_load:
        print("No keywords loaded from file. Exiting.")
        return
    
    # Show preview of what will be loaded
    print(f"\n📋 Preview of keywords to be loaded:")
    for i, keyword in enumerate(keywords_to_load[:5]):  # Show first 5
        keyword_text = keyword.get('keyword', 'N/A')
        category = keyword.get('category', 'N/A')
        print(f"   {i+1}. {keyword_text} [{category}]")
    
    if len(keywords_to_load) > 5:
        print(f"   ... and {len(keywords_to_load) - 5} more keywords")
    
    # Ask for confirmation
    print(f"\n🤔 Ready to load {len(keywords_to_load)} keywords into the database?")
    confirm = input("   Type 'yes' to continue or 'no' to cancel: ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("   ⏸️  Operation cancelled by user.")
        return
    
    db_connection = None
    try:
        print(f"\n🔗 Connecting to database...")
        db_connection = XBotDetectorDB()
        
        print(f"📤 Syncing {len(keywords_to_load)} keywords with the database...")
        results = db_connection.add_keywords_bulk(keywords_to_load)
        
        # Display results
        print(f"\n✅ Sync completed successfully!")
        print(f"   📈 Added: {results.get('added', 0)} new keywords")
        print(f"   ⏭️  Skipped: {results.get('skipped', 0)} duplicates")
        
        # Show database stats
        try:
            stats = db_connection.get_database_stats()
            print(f"\n📊 Current database status:")
            print(f"   Total keywords: {stats['keywords']['total']}")
            print(f"   Active keywords: {stats['keywords']['active']}")
        except Exception as e:
            print(f"   (Could not retrieve database stats: {e})")
        
    except Exception as e:
        print(f"\n❌ An error occurred during database operations: {e}")
        print(f"   💡 Please check your MongoDB connection and credentials")
    finally:
        if db_connection:
            db_connection.close_connection()
            print(f"\n🔌 Database connection closed.")

# FIXED: Correct syntax with underscores, not asterisks
if __name__ == "__main__":
    main()
