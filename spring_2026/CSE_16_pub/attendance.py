import pandas as pd
from datetime import datetime

def process_csv(input_file, output_file):

    # 2. Load the CSV 
    # header=0 handles the removal of the first row (by treating it as the header)
    df = pd.read_csv(input_file)

    # Identify columns by index
    day_col = df.columns[0]
    id_col = df.columns[1]

    # 1. Ask for the filter date with robustness
    print("Example format: 4/6/2026")
    user_input = input("Enter the day to filter by: ").strip()
    
    try:
        # Convert user input to a date object for comparison
        target_date = pd.to_datetime(user_input).date()
    except Exception as e:
        print(f"Error: Could not parse input date. Please use a format like MM/DD/YYYY. Error: {e}")
        return

    


    # 3. Robustly convert the first column to datetime and strip time
    # errors='coerce' turns unparseable dates into NaT (Not a Time) instead of crashing
    df[day_col] = pd.to_datetime(df[day_col], errors='coerce')
    
    # 4. Filter by the target date (comparing only the date component)
    # We use .dt.date to ignore the '15:56:28' time portion
    filtered_df = df[df[day_col].dt.date == target_date].copy()

    # 5. Remove repeats in second column, keep the 'later' one
    # Assuming "later" means the last occurrence in the file
    final_df = filtered_df.drop_duplicates(subset=[id_col], keep='last')

    print(f"number of rows: {final_df.count()}")
    # 6. Output the filtered rows as CSV
    final_df.to_csv(output_file, index=False)
    print(f"Success! Filtered for {target_date}. Results saved to {output_file}")

if __name__ == "__main__":
    # Change these to your actual file names
    input_filename = input("input filename:\n")
    # try:
    process_csv("csvs\\"+input_filename, "processed\\"+input_filename[:-4] + '_processed.csv')
    # except:pass 

    # process_csv(input_filename, "processed\\"+input_filename[:-4] + '_processed.csv')
