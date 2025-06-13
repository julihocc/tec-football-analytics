# Migrating to Python: Soccer Analytics

A Python conversion of Clive Beggs' "Soccer analytics: an introduction using R"

Original R repository by Clive Beggs (16th January 2024)  
Python conversion completed: June 13, 2025

## Overview

This repository contains both the original R code and Python conversions for the examples in the book 'Soccer analytics: an introduction using R' by Clive Beggs. The project demonstrates how to migrate R-based data analysis workflows to Python, maintaining equivalent functionality while leveraging Python's data science ecosystem.

## Repository Structure

- **R code files**: Original R code for each chapter (e.g., `chapter01/chapter01.R`)
- **Python code files**: Converted Python equivalents (e.g., `chapter03/chapter03.py`)
- **Data files**: CSV datasets in the `data/` directory
- **Virtual environment**: Isolated Python environment for dependencies

## Getting Started with Python

### Quick Setup

1. **Automatic setup** (recommended):
   ```bash
   ./setup_venv.sh
   ```

2. **Manual setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running Python Scripts

```bash
# Activate virtual environment
source activate_venv.sh

# Run a converted script
cd chapter03
python chapter03.py
```

### Dependencies

The Python version requires:
- pandas (data manipulation)
- numpy (numerical computing)
- requests (web data access)
- beautifulsoup4 (web scraping)
- matplotlib/seaborn (visualization)

## R vs Python Comparison

The conversion maintains the same analytical workflows while adapting to Python conventions:Analytics
Soccer analytics: an introduction with R repository

Clive Beggs (16th January 2024)

The files contained in the SoccerAnalytics repository accompany the book ‘Soccer analytics: an introduction using R’ by Clive Beggs. There are two types of file in the repository, R code files and csv data files. The code files contain the R code for the examples in the book and are named according to the various chapters, while the csv files contain the data used by the R code in the examples. There are eleven R code files, one for each chapter of the book, and these should run automatically in RStudio. The data required to run the example codes is contained in the csv files.

The R code and csv files should be copied onto the users hard disk, with the csv files stored in a directory where they can be accessed by the R code. In the example codes, it is assumes that the csv data files are located in a directory called ‘Datasets’.

The files contained in the SoccerAnalytics repository are covered by the MIT licence which permits the user to copy, modify, merge and utilise the software free of charge. Under this licence, users should note that:   

“THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.”

| Feature | R | Python |
|---------|---|--------|
| Data Frames | `data.frame()` | `pd.DataFrame()` |
| Missing Values | `NA`, `na.rm=TRUE` | `np.nan`, `skipna=True` |
| Statistics | `psych::describeBy()` | Custom `describe_data()` |
| Web Scraping | `rvest` | `requests` + `BeautifulSoup` |
| Plotting | `ggplot2` | `matplotlib`/`seaborn` |

## Converted Chapters

✅ **Chapter 3**: Complete conversion with all examples  
- Arsenal data analysis
- Missing data handling  
- Web data loading
- Descriptive statistics
- Web scraping

🚧 **Other chapters**: Available in R, Python conversion in progress

## Virtual Environment Management

The project uses a Python virtual environment to ensure clean dependency management:

- **Setup**: `./setup_venv.sh` - Creates and configures the environment
- **Activate**: `source activate_venv.sh` - Quick activation
- **Manual**: `source venv/bin/activate` - Standard activation
- **Deactivate**: `deactivate` - Exit the environment

## File Organization

```
migrating-to-python/
├── venv/                 # Python virtual environment
├── setup_venv.sh         # Automated setup script
├── activate_venv.sh      # Quick activation script
├── requirements.txt      # Python dependencies
├── data/                 # CSV datasets
├── chapter01/            # R code only
├── chapter02/            # R code only  
├── chapter03/            # R + Python versions
│   ├── chapter03.R       # Original R script
│   ├── chapter03.py      # Python conversion
│   └── README.md         # Chapter-specific documentation
└── chapter04-11/        # R code only
```

## License

The files contained in the SoccerAnalytics repository are covered by the MIT licence which permits the user to copy, modify, merge and utilise the software free of charge. Under this licence, users should note that:   

“THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.”
