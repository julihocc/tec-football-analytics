# Chapter 2: Basic Statistical Analysis in R

## 1. Basic Data Manipulation (Example 2.1)

Introduction to creating and manipulating vectors in R.

```r
# Create basic match data vectors
MatchID <- c(1:10)  # Match identifiers
GoalsFor <- c(0,2,4,1,3,0,2,2,3,1)   # Goals scored
GoalsAgainst <- c(1,1,3,3,0,0,1,1,1,0)  # Goals conceded
GoalDiff <- GoalsFor-GoalsAgainst   # Goal differences

# Calculate basic statistics for Goals Scored
mean(GoalsFor)  # Mean
median(GoalsFor)  # Median
sd(GoalsFor)  # Standard deviation
var(GoalsFor)  # Variance

# Create data frame
goals_dat <- cbind.data.frame(MatchID, GoalsFor, GoalsAgainst, GoalDiff)
print(goals_dat)
```

Key concepts:

- Vector creation using `c()`
- Basic statistical functions (`mean()`, `median()`, `sd()`, `var()`)
- Data frame creation with `cbind.data.frame()`

## 2. Data Frame Operations (Example 2.2)

Exploring data frame structure and manipulation.

```r
# Data frame inspection
names(goals_dat)  # Column names
head(goals_dat, 8)  # First 8 rows
nrow(goals_dat)  # Number of rows
ncol(goals_dat)  # Number of columns
dim(goals_dat)  # Dimensions
str(goals_dat)  # Structure

# Column selection methods
goals_dat$GoalsFor  # Using $ operator
goals_dat[,2]  # Using index
goals_dat[,c(2,3)]  # Multiple columns
goals_dat[c(3,4,5),]  # Row selection
```

Key concepts:

- Data frame inspection functions
- Column and row selection methods
- Structure analysis

## 3. Conditional Operations (Example 2.3)

Creating match outcomes using conditional logic.

```r
# Method 1: Using ifelse
outcome1 <- ifelse(goals_dat$GoalsFor > goals_dat$GoalsAgainst, "Win", "Did not win")

# Method 2: Using for loop and if statements
outcome2 <- c()
n <- nrow(goals_dat)  
for(i in 1:n){
  if(goals_dat$GoalsFor[i] > goals_dat$GoalsAgainst[i]){outcome2[i] <- "Win"}
  if(goals_dat$GoalsFor[i] < goals_dat$GoalsAgainst[i]){outcome2[i] <- "Lose"}
  if(goals_dat$GoalsFor[i] == goals_dat$GoalsAgainst[i]){outcome2[i] <- "Draw"}
}

# Add results to data frame
match_dat <- cbind.data.frame(goals_dat, outcome2)
colnames(match_dat)[colnames(match_dat) == 'outcome2'] <- 'Result'
```

Key concepts:

- Conditional logic with `ifelse()`
- For loops and if statements
- Column renaming in data frames

## 4. Descriptive Statistics (Example 2.4-2.5)

Advanced statistical analysis using the psych package.

```r
# Basic summary
summary(match_dat)

# Using psych package
library(psych)
des_res <- describeBy(match_dat[,c(2:4)])
print(des_res)

# Reading external data
dat <- read.csv("Arsenal_Chelsea_comparison.csv")
des_results <- describeBy(dat[,c(2:7)])
```

Key concepts:

- Summary statistics
- Using external packages (psych)
- Reading CSV files
- Advanced descriptive statistics

## 5. Data Visualization (Example 2.6-2.8)

Creating various plots for football data analysis.

```r
# Time series plot
seasons <- c("2011","2012","2013","2014","2015","2016","2017","2018","2019","2020")
plot(seasons, dat$Arsenal_GF, type="o", lty=1, pch=20, col="black", ylim=c(0,140), 
     ylab="Goals", xlab="Season")
lines(seasons, dat$Chelsea_GF, type="o", lty=2, pch=20)

# Box plot
boxplot(dat[,c(2,3,5,6)], ylab="Goals")

# Scatter plot with regression lines
plot(dat$Chelsea_GA, dat$Chelsea_points, pch=20, col="black", xlim=c(0,60), 
     ylim=c(0,100), ylab="Points", xlab="Goals conceded")
abline(lm(dat$Chelsea_points ~ dat$Chelsea_GA), lty=1)
```

Key concepts:

- Time series plotting
- Box plots for comparison
- Scatter plots with regression lines
- Plot customization

## 6. Statistical Testing (Example 2.9)

Performing statistical tests on football data.

```r
# Paired t-tests
t.test(dat$Arsenal_GF, dat$Chelsea_GF, paired=TRUE) 
t.test(dat$Arsenal_GA, dat$Chelsea_GA, paired=TRUE)

# Correlation tests
cor.test(dat$Arsenal_GA, dat$Arsenal_points)
cor.test(dat$Chelsea_GA, dat$Chelsea_points)
```

Key concepts:

- Paired t-tests for team comparison
- Correlation analysis
- Statistical inference
- P-value interpretation

---

## Summary of Key R Functions

1. Data Management:
   - `c()` for vector creation
   - `cbind.data.frame()` for data frame creation
   - `read.csv()` for file import

2. Basic Statistics:
   - `mean()`, `median()`, `sd()`, `var()`
   - `summary()`
   - `describeBy()` from psych package

3. Data Visualization:
   - `plot()` for various plot types
   - `boxplot()` for distribution comparison
   - `abline()` for regression lines

4. Statistical Testing:
   - `t.test()` for comparing means
   - `cor.test()` for correlation analysis

5. Data Manipulation:
   - `ifelse()` for conditional operations
   - Index-based selection
   - For loops and if statements
