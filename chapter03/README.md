# Chapter 3: R to Python Migration

This directory contains both the original R script and its Python equivalent for Chapter 3 data analysis examples.

## Files

- `chapter03.R` - Original R script with portable path handling
- `chapter03.py` - Python conversion of the R script
- `../requirements.txt` - Python package dependencies

## Python Dependencies

Install the required packages using:

```bash
pip install -r ../requirements.txt
```

Or install individual packages:

```bash
pip install pandas numpy requests beautifulsoup4 lxml
```

## Key Differences Between R and Python Versions

### 1. Data Frame Operations
- **R**: `data.frame()`, `names()`, `str()`, `head()`
- **Python**: `pd.DataFrame()`, `.columns`, `.info()`, `.head()`

### 2. Missing Data Handling
- **R**: `NA`, `na.rm = TRUE`, `na.omit()`
- **Python**: `np.nan`, `skipna=True`, `.dropna()`

### 3. Statistical Functions
- **R**: `describeBy()` from psych package
- **Python**: Custom `describe_data()` function using pandas `.describe()` + additional stats

### 4. Web Scraping
- **R**: `rvest` package with `read_html()`, `html_table()`
- **Python**: `requests` + `BeautifulSoup` + `pd.read_html()`

### 5. Data Import/Export
- **R**: `read.csv()`, `write.csv()`
- **Python**: `pd.read_csv()`, `.to_csv()`

## Features Implemented

✅ **Example 3.1**: Arsenal data analysis with calculated variables (Goal Difference, Total Goals, Shot Ratios)

✅ **Example 3.2**: Data filtering and splitting by match results (Wins, Losses, Draws)

✅ **Example 3.3**: Missing data handling with comprehensive descriptive statistics

✅ **Example 3.4**: Loading data from internet sources (Football-Data.co.uk)

✅ **Example 3.5**: Multi-season data download and concatenation

✅ **Web Scraping Example**: Wikipedia table scraping (equivalent to rvest functionality)

## Notes on worldfootballR Conversion

The original R script uses the `worldfootballR` package for advanced football data analysis (Examples 3.6-3.8). Since this is an R-specific package, the Python version includes:

1. **Alternative approaches** for similar functionality
2. **Direct web scraping** methods using requests/BeautifulSoup
3. **Recommendations** for Python football data packages

### Suggested Python Alternatives for worldfootballR:

- **Direct scraping**: FBref, ESPN, or other sports sites
- **APIs**: Official league APIs, RapidAPI sports endpoints
- **Python packages**: `football-data-api`, `soccerdata`, or similar

## Running the Scripts

### R Version:
```bash
cd chapter03
Rscript chapter03.R
```

### Python Version:
```bash
cd chapter03
python3 chapter03.py
```

Both scripts are designed to work from any directory within the project structure and will automatically locate the required data files.

## Output Comparison

The Python version produces equivalent output to the R version for all implemented examples:

- Data loading and inspection
- Statistical calculations
- Data transformations
- Web data retrieval
- Descriptive statistics
- Data filtering and grouping

The descriptive statistics in Python closely match R's `psych::describeBy()` output, including mean, standard deviation, median, MAD, min/max, range, skewness, kurtosis, and standard error.
