#!/usr/bin/env python3
"""
Chapter 3: Python code
Converted from R script by Clive Beggs 7th March 2023
Python conversion date: June 13, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

def ensure_correct_directory():
    """
    Ensure we're in the correct directory for relative paths to work.
    This approach makes the script portable and works from any directory.
    """
    current_dir = os.getcwd()
    
    # If not in chapter03, look for it relative to current directory
    if os.path.basename(current_dir) != "chapter03":
        # Try various possible locations for chapter03
        possible_dirs = [
            "chapter03",                                    # ./chapter03
            os.path.join("migrating-to-python", "chapter03"), # ./migrating-to-python/chapter03
            os.path.join("..", "chapter03"),                   # ../chapter03
            os.path.join(".", "chapter03")                     # ./chapter03 (redundant but safe)
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.exists(os.path.join(dir_path, "chapter03.py")):
                os.chdir(dir_path)
                # Only show message if we actually changed directories
                if os.path.basename(current_dir) != "chapter03":
                    print(f"Changed working directory to: {os.getcwd()}")
                break
    
    # Verify we can find the data file
    data_file = "../data/Arsenal_home_2020.csv"
    if not os.path.exists(data_file):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for data file at: {data_file}")
        raise FileNotFoundError("Data file not found. Please ensure the data directory exists relative to chapter03.")
    
    return data_file

def clear_variables():
    """
    Clear variables from the current namespace (equivalent to R's rm(list=ls()))
    """
    # Get all variable names in the current namespace
    current_vars = list(globals().keys())
    # Keep only built-in variables, imported modules, and functions
    protected_vars = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__annotations__', '__builtins__',
                     'os', 'sys', 'pd', 'np', 'requests', 'BeautifulSoup', 'warnings',
                     'ensure_correct_directory', 'clear_variables', 'describe_data']
    
    for var in current_vars:
        if var not in protected_vars and not var.startswith('_'):
            if var in globals():
                del globals()[var]

def describe_data(df, columns=None):
    """
    Equivalent to R's describeBy function from psych package
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    desc_stats = df[columns].describe()
    
    # Add additional statistics similar to R's describeBy
    additional_stats = {}
    for col in columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            additional_stats[col] = {
                'vars': 1,
                'n': len(col_data),
                'mean': col_data.mean(),
                'sd': col_data.std(),
                'median': col_data.median(),
                'trimmed': col_data.mean(),  # Simplified - would need trimmed mean calculation
                'mad': col_data.sub(col_data.median()).abs().median(),
                'min': col_data.min(),
                'max': col_data.max(),
                'range': col_data.max() - col_data.min(),
                'skew': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
                'se': col_data.std() / np.sqrt(len(col_data))
            }
    
    print("\nDescriptive Statistics:")
    print("=" * 80)
    for col in columns:
        if col in additional_stats:
            stats = additional_stats[col]
            print(f"\n{col}:")
            print(f"  vars: {stats['vars']}")
            print(f"  n: {stats['n']}")
            print(f"  mean: {stats['mean']:.2f}")
            print(f"  sd: {stats['sd']:.2f}")
            print(f"  median: {stats['median']:.2f}")
            print(f"  mad: {stats['mad']:.2f}")
            print(f"  min: {stats['min']:.2f}")
            print(f"  max: {stats['max']:.2f}")
            print(f"  range: {stats['range']:.2f}")
            print(f"  skew: {stats['skew']:.2f}")
            print(f"  kurtosis: {stats['kurtosis']:.2f}")
            print(f"  se: {stats['se']:.2f}")

def main():
    """Main function to run all examples"""
    
    # Ensure we're in the correct directory
    data_file = ensure_correct_directory()
    
    print("="*60)
    print("Chapter 3: Python Data Analysis Examples")
    print("="*60)
    
    # ========================================
    # Code for Example 3.1
    # ========================================
    print("\n" + "="*40)
    print("Example 3.1: Arsenal Data Analysis")
    print("="*40)
    
    # Clear variables (equivalent to rm(list = ls()))
    clear_variables()
    
    # Load data in form of CSV file using relative path
    ArsenalHome = pd.read_csv("../data/Arsenal_home_2020.csv")
    print("\nArsenal Home Data:")
    print(ArsenalHome)
    
    # Inspect data
    print(f"\nColumn names: {list(ArsenalHome.columns)}")  # Equivalent to names()
    print(f"\nData structure:")  # Equivalent to str()
    print(ArsenalHome.info())
    print(f"\nData types:\n{ArsenalHome.dtypes}")
    
    Ar_dat = ArsenalHome.copy()  # This makes a working copy of the data frame
    
    # Step 1 – Create the new empty variables populated with NAs
    Ar_dat["GD"] = np.nan  # Goal difference
    Ar_dat["TG"] = np.nan  # Total goals scored
    Ar_dat["HTSR"] = np.nan  # Home team shots ratio
    Ar_dat["ATSR"] = np.nan  # Away team shots ratio
    
    # Step 2 - Populate the new columns with the calculated values
    Ar_dat["GD"] = Ar_dat["HG"] - Ar_dat["AG"]  # Goal difference
    Ar_dat["TG"] = Ar_dat["HG"] + Ar_dat["AG"]  # Total goals
    Ar_dat["HTSR"] = round(Ar_dat["HS"] / (Ar_dat["HS"] + Ar_dat["AS"]), 3)  # HTSR rounded to 3dp
    Ar_dat["ATSR"] = round(Ar_dat["AS"] / (Ar_dat["AS"] + Ar_dat["HS"]), 3)  # ATSR rounded to 3dp
    
    print(f"\nUpdated column names: {list(Ar_dat.columns)}")
    print(f"\nFirst 8 rows of modified data:")
    print(Ar_dat.head(8))
    
    # Export results as CSV file (commented out like in R)
    # Ar_dat.to_csv("Arsenal_home_shots_ratio.csv", index=False)
    
    # ========================================
    # Code for Example 3.2
    # ========================================
    print("\n" + "="*40)
    print("Example 3.2: Data Splitting by Result")
    print("="*40)
    
    # Inspect data
    print("\nFirst 6 rows of data:")
    print(Ar_dat.head(6))
    
    # This splits the data into separate win, lose and draw data frames
    win = Ar_dat[Ar_dat["Result"] == "W"].copy()  # Wins
    lose = Ar_dat[Ar_dat["Result"] == "L"].copy()  # Losses
    draw = Ar_dat[Ar_dat["Result"] == "D"].copy()  # Draws
    
    # Display sub-groups
    print("\nWins:")
    print(win)
    print("\nLosses:")
    print(lose)
    print("\nDraws:")
    print(draw)
    
    # ========================================
    # Code for Example 3.3
    # ========================================
    print("\n" + "="*40)
    print("Example 3.3: Handling Missing Data")
    print("="*40)
    
    # Create data with some missing data entries
    players = ["Paul", "Stephen", "James", "Kevin", "Tom", "Edward", "John", "David"]
    shots = [2.4, 3.6, 0.3, 1.1, 4.2, 2.3, np.nan, 0.6]  # Average shots per game
    goals = [0.2, 0.6, 0.0, 0.1, 0.7, 0.3, 0.1, 0.0]  # Average goals per game
    passes = [23.1, np.nan, 39.2, 25.5, 18.6, 37.4, 28.3, 28.3]  # Average passes per game
    tackles = [6.3, 4.5, 10.6, 9.8, 4.1, 5.3, 11.2, 7.8]  # Average tackles per game
    
    # Create data frame
    perf_dat = pd.DataFrame({
        'players': players,
        'shots': shots,
        'goals': goals,
        'passes': passes,
        'tackles': tackles
    })
    print("\nPerformance data with missing values:")
    print(perf_dat)
    
    # Completely remove lines containing NAs
    print("\nData with NAs removed:")
    print(perf_dat.dropna())
    
    # Try out 'mean' function on data
    print(f"\nMean calculations:")
    print(f"Mean shots (with NaN): {perf_dat['shots'].mean()}")  # This returns NaN
    print(f"Mean goals (no NaN): {perf_dat['goals'].mean()}")  # This works
    print(f"Mean passes (with NaN): {perf_dat['passes'].mean()}")  # This returns NaN
    print(f"Mean tackles (no NaN): {perf_dat['tackles'].mean()}")  # This works
    
    # Now with NaN handling (equivalent to na.rm = TRUE)
    print(f"\nMean calculations (ignoring NaN):")
    print(f"Mean shots (ignoring NaN): {perf_dat['shots'].mean(skipna=True)}")
    print(f"Mean goals (ignoring NaN): {perf_dat['goals'].mean(skipna=True)}")
    print(f"Mean passes (ignoring NaN): {perf_dat['passes'].mean(skipna=True)}")
    print(f"Mean tackles (ignoring NaN): {perf_dat['tackles'].mean(skipna=True)}")
    
    # Descriptive statistics (equivalent to describeBy from psych package)
    describe_data(perf_dat, ['shots', 'goals', 'passes', 'tackles'])
    
    # ========================================
    # Code for Example 3.4
    # ========================================
    print("\n" + "="*40)
    print("Example 3.4: Loading Data from Internet")
    print("="*40)
    
    # Clear existing variables and data from the workspace
    clear_variables()
    
    # Load data from Internet (first 380 rows, first 16 columns)
    url = 'https://www.football-data.co.uk/mmz4281/2021/E0.csv'
    try:
        EPL2020_dat = pd.read_csv(url).head(380).iloc[:, :16]
        
        # Inspect data
        print(f"Column names: {list(EPL2020_dat.columns)}")
        print(f"\nFirst 10 rows:")
        print(EPL2020_dat.head(10))
        
        # Export data (commented out like in R)
        # EPL2020_dat.to_csv("EPL_results_2021.csv", index=False)
        
    except Exception as e:
        print(f"Error loading data from internet: {e}")
        print("Continuing with next example...")
    
    # ========================================
    # Code for Example 3.5
    # ========================================
    print("\n" + "="*40)
    print("Example 3.5: Multiple Seasons Data Download")
    print("="*40)
    
    # Clear existing variables and data from the workspace
    clear_variables()
    
    # Download results from website for the 5 seasons 2016-2021
    seasons = ["1617", "1718", "1819", "1920", "2021"]
    division = ["E0"] * 5  # "E0" is the EPL
    
    # Create URLs
    urls = [f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv" 
            for season, div in zip(seasons, division)]
    
    # Load all the data using a loop and selecting just a few variables
    download_data = []
    columns_to_keep = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    
    try:
        for url in urls:
            print(f"Loading data from: {url}")
            temp = pd.read_csv(url)
            temp = temp[columns_to_keep]
            download_data.append(temp)
        
        # Combine all data
        download_data = pd.concat(download_data, ignore_index=True)
        
        # Inspect data frame
        print(f"\nFirst 10 rows:")
        print(download_data.head(10))
        print(f"\nLast 10 rows:")
        print(download_data.tail(10))
        print(f"\nTotal rows: {len(download_data)}")
        
    except Exception as e:
        print(f"Error loading multiple seasons data: {e}")
        print("Continuing with next example...")
    
    # ========================================
    # Code for Example 3.6-3.8: Football Data Analysis
    # ========================================
    print("\n" + "="*40)
    print("Examples 3.6-3.8: Advanced Football Data Analysis")
    print("="*40)
    
    print("\nNote: Examples 3.6-3.8 use the worldfootballR package which is R-specific.")
    print("For Python equivalents, you would typically use:")
    print("1. Direct web scraping with requests/BeautifulSoup")
    print("2. APIs like FBref, ESPN, or other sports data providers")
    print("3. Specialized Python packages like football-data-api or similar")
    print("\nHere's an example of web scraping similar to the rvest example:")
    
    # ========================================
    # Web Scraping Example (equivalent to rvest)
    # ========================================
    print("\n" + "="*40)
    print("Web Scraping Example: Transfer Windows")
    print("="*40)
    
    try:
        # Read HTML from Wikipedia
        url = "https://en.wikipedia.org/wiki/Transfer_window"
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the first table with class 'wikitable'
        table = soup.find('table', {'class': 'wikitable'})
        
        if table:
            # Convert table to pandas DataFrame
            tw_df = pd.read_html(str(table), header=0)[0]
            
            print("\nTransfer Window Data:")
            print(tw_df)
        else:
            print("No wikitable found on the page")
            
    except Exception as e:
        print(f"Error in web scraping: {e}")
    
    print("\n" + "="*60)
    print("Python conversion completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
