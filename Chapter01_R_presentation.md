# Chapter 1: Statistical Analysis in Soccer

## 1. EPL Champions Analysis (1992-2021)

This section demonstrates basic data visualization in R using Premier League champions data.

```r
# Create vectors for clubs and their titles
Clubs <- c("Arsenal","Blackburn Rovers","Chelsea","Leicester City","Liverpool","Man City","Man United")
Titles <- c(3,1,5,1,1,6,13)
epl.dat <- cbind.data.frame(Clubs,Titles)
print(epl.dat)

# Bar plot
barplot(epl.dat$Titles, names.arg=epl.dat$Clubs, ylim=c(0,14))
title("Bar chart of EPL champions 1992-2021")

# Pie chart
colours = gray.colors(length(epl.dat$Clubs))
pie(Titles, labels = Clubs , col=colours,main="Pie Chart of EPL champtions 1992-2021")
```

Key concepts:

- Data organization using vectors and data frames
- Basic bar plot visualization using `barplot()`
- Pie chart representation with grayscale colors using `pie()`
- Data combination using `cbind.data.frame()`

## 2. Manager Performance Comparison

Statistical comparison of Manchester United managers using Chi-square test.

```r
# Create vectors for manager statistics
Manager <- c("Ferguson","Moyes","van Gaal","Mourinho","Solskjaer","Rangnick")
Wins <- c(895,27,54,84,91,11)
Draws <- c(338,9,25,32,37,10)
Losses <- c(267,15,24,28,40,8)

mu.record <- cbind.data.frame(Manager,Wins,Draws,Losses)
print(mu.record)

# Compare Ferguson vs Rangnick
man1 <- "Ferguson"
man2 <- "Rangnick"

Manager1 <- mu.record[mu.record$Manager == man1,]   
Manager2 <- mu.record[mu.record$Manager == man2,]   

# Create a contingency table
temp = as.matrix(rbind(Manager1[,c(2:4)], Manager2[,c(2:4)]))
ContTab <- as.table(temp)
dimnames(ContTab) = list(Manager = c("Manager 1", "Manager 2"),
                         Outcome = c("Wins","Draws", "Loses"))
print(ContTab)

# Perform Chi-square test
chsqRes = chisq.test(ContTab)
print(chsqRes)
```

Key insights:

- Comprehensive comparison between different managers (e.g., Ferguson vs Rangnick)
- Analysis of Wins, Draws, and Losses using contingency tables
- Statistical significance testing using Chi-square test
- Data subsetting and manipulation techniques

## 3. Points vs Shots Analysis (2020-21 Season)

Linear regression analysis examining the relationship between shots taken and points achieved.

```r
# Create data vectors
Points <- c(61, 55, 41, 39, 67, 44, 59, 28, 59, 66, 69, 86, 74, 45, 23, 43, 62, 26, 65, 45)
Shots <- c(455,518,476,383,553,346,395,440,524,472,600,590,517,387,319,417,442,336,462,462)

dat2020 <- cbind.data.frame(Points, Shots)
head(dat2020)  # Display first 6 rows

# Build OLS regression model
mod2020 <- lm(Points ~ Shots, data = data.frame(dat2020))
summary(mod2020)

# Create scatter plot with best-fit line
plot(dat2020$Shots, dat2020$Points, pch=20, col="black", xlim=c(0,800), 
     ylim=c(0,100), ylab="Points", xlab="Shots")
abline(lm(dat2020$Points ~ dat2020$Shots), lty=1)

# Predictions for 2021-22 season
# Chelsea (585 shots)
ChelPts.2021 <- predict(mod2020, list(Shots=585))
print(ChelPts.2021)

# Manchester City (704 shots)
MCPts.2021 <- predict(mod2020, list(Shots=704))
print(MCPts.2021)

# Norwich City (374 shots)
NCPts.2021 <- predict(mod2020, list(Shots=374))
print(NCPts.2021)
```

Key concepts:

- Linear regression modeling using `lm()`
- Predictive analytics with `predict()`
- Data visualization with scatter plots and best-fit lines
- Model application to real-world predictions

## 4. Betting Odds Analysis

Analysis of betting odds from different bookmakers (William Hill and Pinnacle).

```r
# William Hill match odds
wh_hwodds <- 2.15 # Odds for home win
wh_dodds <- 3.30 # Odds for draw
wh_awodds <- 3.50 # Odds for away win

# Pinnacle match odds
p_hwodds <- 2.13 # Odds for home win
p_dodds <- 3.61 # Odds for draw
p_awodds <- 3.64 # Odds for away win

# Compile odds into data frame
WH_odds <- c(wh_hwodds,wh_dodds,wh_awodds)
Pin_odds <- c(p_hwodds,p_dodds,p_awodds)
bet.dat <- cbind.data.frame(WH_odds,Pin_odds)

# Calculate implied probabilities
bet.dat$WH_prob <- round(1/bet.dat$WH_odds,3)
bet.dat$Pin_prob <- round(1/bet.dat$Pin_odds,3)
rownames(bet.dat) <- c("Home win","Draw","Away win")
print(bet.dat)

# Calculate over-rounds
WH_or <- sum(bet.dat$WH_prob)-1
print(WH_or)  # William Hill's over-round

Pin_or <- sum(bet.dat$Pin_prob)-1
print(Pin_or)  # Pinnacle's over-round

# Calculate potential profit
wager <- 10 # £10 wager with WH on Tottenham to win
profit <- wager * (wh_hwodds-1)
print(profit)
```

Concepts covered:

- Odds representation and manipulation
- Implied probability calculation
- Over-round calculation (bookmaker's margin)
- Profit calculation on potential wagers
- Comparison of different bookmakers' odds

## 5. Cup Draw Simulation

Demonstration of tournament draw simulation using random sampling in R.

```r
# Create vector of team numbers
teams <- c(1:16)
print(teams)

# Create matrix to store results
cup.draw <- as.data.frame(matrix(0,8,2))
colnames(cup.draw) <- c("HomeTeam", "AwayTeam")

# Randomly sample eight home teams
set.seed(123) # Makes draw results repeatable
samp.HT <- sample(teams, size=8, replace=FALSE)
samp.AT <- sample(teams[-samp.HT])

# Assign teams to draw
cup.draw$HomeTeam <- samp.HT
cup.draw$AwayTeam <- samp.AT
print(cup.draw)
```

Key features:

- 16-team tournament simulation
- Random assignment of home/away teams using `sample()`
- Matrix manipulation for results storage
- Reproducible results using `set.seed()`
- Data frame creation and column naming

---

## Summary of Key R Concepts Covered

1. Data Structures:
   - Vectors (`c()`)
   - Data frames (`data.frame()`, `cbind.data.frame()`)
   - Matrices (`matrix()`)
   - Tables (`as.table()`)

2. Statistical Analysis:
   - Chi-square testing (`chisq.test()`)
   - Linear regression (`lm()`)
   - Predictive modeling (`predict()`)

3. Data Visualization:
   - Bar plots (`barplot()`)
   - Pie charts (`pie()`)
   - Scatter plots (`plot()`)
   - Best fit lines (`abline()`)

4. Data Management:
   - Random sampling (`sample()`)
   - Matrix manipulation
   - Data frame operations
   - Probability calculations

5. Applied Mathematics:
   - Sports betting calculations
   - Statistical inference
   - Probability analysis
   - Tournament simulations
