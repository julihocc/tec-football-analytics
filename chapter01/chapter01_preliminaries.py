# Python Preliminaries for Soccer Data Analytics (Chapter 1)
# -------------------------------------------------------------
# This script introduces the essential Python concepts and libraries needed to follow the analyses in Chapter 1.
# It is designed for high school students who are new to data analytics and Python programming.
#
# The goal is to help you understand the basics so you can confidently run, modify, and learn from the code in this chapter.

# %%
# 1. What is Python?
# Python is a popular programming language used for data analysis, science, and many other fields.
# It is known for being easy to read and learn, making it a great choice for beginners.

# %%
# 2. How do we use Python for data analytics?
# In this course, we use Python to:
# - Organize and explore data (like soccer match results)
# - Visualize data with charts and graphs
# - Perform calculations and statistical tests
# - Make predictions using simple models

# %%
# 3. Python Libraries: The "Toolbox" for Data Analytics
# Python has special packages (called "libraries") that make data analysis easier.
# For Chapter 1, we use these main libraries:

# Import the libraries (run this cell first in your notebook or script)
import pandas as pd      # For working with tables of data (DataFrames)
import matplotlib.pyplot as plt  # For making charts and plots
import numpy as np       # For calculations and working with numbers
from scipy.stats import chi2_contingency  # For statistical tests

# %%
# 4. What does each library do?
# - pandas: Lets us organize data in tables (like Excel, but in Python)
# - matplotlib: Lets us draw bar charts, pie charts, and more
# - numpy: Helps with math, especially with lists of numbers
# - scipy.stats: Lets us do statistical tests (like the chi-square test)

# %%
# 5. How do I run Python code?
# - You can run each section (cell) one at a time in VS Code's interactive window or a Jupyter notebook.
# - The code after the # %% marker is a "cell". Click in the cell and press Shift+Enter to run it.
# - You can change numbers or try your own data to see what happens!

# %%
# 6. Tips for Success
# - Read the comments (lines starting with #) to understand what each part does.
# - If you see an error, check that you ran the cell that imports the libraries first.
# - Don't be afraid to experiment—change values and see how the results change!

# Now you're ready to start exploring soccer data with Python!
# Move on to Chapter01.py to see real examples and try them yourself.
