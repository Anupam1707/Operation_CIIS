# keyword_manager.py (Version 2.0 - More flexible)
import sys
import csv
import json
from storage import XBotDetectorDB  # Correctly import the class from 'storage.py'

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
    
    # Check if a command-line argument (the file path) was provided
    if len(sys.argv) < 2:
        print("\n❌ Error: You must provide the path to the data file.")
        print("   Usage: python keyword_manager.py data/keywords.json")
        print("   Usage: python keyword_manager.py data/keywords.csv\n")
        return  # Exit the script
    
    # The file path is the first argument after the script name
    input_file_path = sys.argv[1]
    
    keywords_to_load = load_keywords_from_file(input_file_path)
    
    if not keywords_to_load:
        print("No keywords loaded from file. Exiting.")
        return
    
    db_connection = None
    try:
        db_connection = XBotDetectorDB()
        print(f"\nSyncing {len(keywords_to_load)} keywords with the database...")
        results = db_connection.add_keywords_bulk(keywords_to_load)
        
        # Fixed: Use correct keys from the results dictionary
        print(f"-> Done. Added: {results.get('added', 0)}, Skipped: {results.get('skipped', 0)}.")
        
    except Exception as e:
        print(f"An error occurred during database operations: {e}")
    finally:
        if db_connection:
            db_connection.close_connection()
            print("\nDatabase connection closed.")

# Fixed: Correct syntax with underscores, not asterisks
if __name__ == "__main__":
    main()
