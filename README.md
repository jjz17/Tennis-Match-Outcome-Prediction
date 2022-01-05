# Tennis Match Outcome Prediction Model
This is an individual project created through the Project Track of the [Snowball Initiative](https://dataclub.northeastern.edu/snowball/) at [Northeastern Data Club](http://www.https://dataclub.northeastern.edu/).

#### -- Project Status: Active

## Project Description
* Created a tool that predicts tennis match outcomes (mean accuracy ~ 0.8276) to help players, coaches, and fans to better understand factors that may influence win likelihood
* Engineered features from the play-by-play records of first sets to quantify player performance
* Performed recursive feature selection to separate most relevant features
* Optimized Logistic Regression, kNN, Decision Trees, Naive Bayes, Linear SVM using GridsearchCV to reach the best model. 
* Built a client facing API using streamlit

## Purpose/Objective
The purpose of this project is to build a Classification Machine Learning model that can predict the outcome (winner) of a tennis match, given the play-by-play data of the first set. The target applications of this model are widespread: it can be leveraged as an informative resource for sports-betting, a guide to players and coaches on the improvements that will maximize winning potential, and much more. While the model is currently trained on data from professional men's tennis matches, it can be expanded in the future for compatibility with the women's tour and even for recreational/casual players.

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, streamlit, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Dataset GitHub:** https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv  
**Streamlit Productionization:** https://github.com/dataprofessor/code/tree/master/streamlit/part7

## Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling

## Data Collection and Cleaning
I acquired match data from [Jeff Sackmann's Github](https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv).
\
\
After downloading the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Removed matches from Wimbledon's Final Round Qualifying (they have different rules)
* Parsed the 'play-by-play' strings into useful statistics with custom functions that extract relevant data (points, aces, breaks).
* Feature engineered momentum to quantify trends not explicit in the data (based on consecutive points won)
* Made a new column for points scored for each player
* Made a new column for momentum accumulated for each player
* Made a new column for breaks won for each player
* Made a new column for aces served for each player
* Resulted with 925 samples and 23 features.

## EDA
I looked at the distributions of the data and the value counts for the various quantitative and categorical variables. Below are a few highlights from my analysis. 

![alt text](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/blob/main/visualizations/s1_s2_points_win_relplot.png "First Set Points Colored By Winner")
![alt text](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/blob/main/visualizations/correlation_heatmap.png "Correlations")
![alt text](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/blob/main/visualizations/s1_s2_points_histogram.png "First Set Points Distribution")

## Model Building 

First, I split the data into train and tests sets with a test size of 25%.   

I tried five different models and evaluated them using mean accuracy. I chose mean accuracy because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 11.22
*	**Linear Regression**: MAE = 18.86
*	**Ridge Regression**: MAE = 19.67

## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 

[comment]: <> (### The Problem)

[comment]: <> (The outcomes of tennis matches are notoriously difficult to predict, due to the volatile nature of the sport: changes in momentum, effects from the audience, and a variety of other factors all contribute to its unpredictability. In this project, I tackle this historical challenge, by building a Machine Learning model that predicts the outcome of tennis matches solely based upon play-by-play data from the first set. )

[comment]: <> (### The Data)

[comment]: <> (I acquired match data from [Jeff Sackmann's Github]&#40;https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv&#41;. To parse the 'play-by-play' strings into useful statistics, I wrote a family of functions that extract relevant data &#40;points, aces, breaks&#41;. Furthermore, I performed feature engineering by quantifying the 'momentum' of a player based on consecutive points won. Through the process of data acquisition, cleaning, and wrangling, I ended up with 925 samples and 23 features. )

[comment]: <> (\)

[comment]: <> (\)

[comment]: <> (To do this, I plan on studying and normalizing tennis match data to discover particular features, and to engineer features of my own, which will be fed into a machine learning algorithm to have the match outcomes predicted. I will be looking to discover which variables are most indicative of the match outcomes, and to attempt to create new variables from the existing ones which will further aid the machine learning models. I also plan on testing the data with a variety of different models to determine which one produces the most accurate results.)

[comment]: <> (\)

[comment]: <> (\)

[comment]: <> (I will be testing a variety of classification algorithms, including Logistic Regression, k-Nearest Neighbors, etc.)

[comment]: <> (\)

[comment]: <> (\)

[comment]: <> (&#40;Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modeling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here&#41;)

## Project Structure

- raw dataset and preprocessed dataset are included under the [data](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/data) directory
- model and scaler objects are included under the [models](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/models) directory
- Jupyter Notebook work is included under the [notebooks](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/notebooks) directory
- all source code (data wrangling, exploratory data analysis, model building, custom functions) is included under the [src](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/src) directory
- produced visualizations are included under the [visualizations](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/visualizations) directory

[comment]: <> (## Needs of this project)

[comment]: <> (- frontend developers)

[comment]: <> (- data exploration/descriptive statistics)

[comment]: <> (- data processing/cleaning)

[comment]: <> (- statistical modeling)

[comment]: <> (- writeup/reporting)

[comment]: <> (- etc. &#40;be as specific as possible&#41;)

[comment]: <> (## Getting Started)

[comment]: <> (1. Clone this repo &#40;for help see this [tutorial]&#40;https://help.github.com/articles/cloning-a-repository/&#41;&#41;.)

[comment]: <> (2. Raw Data is being kept [here]&#40;Repo folder containing raw data&#41; within this repo.)

[comment]: <> (    *If using offline data mention that and how they may obtain the data from the froup&#41;*)
    
[comment]: <> (3. Data processing/transformation scripts are being kept [here]&#40;Repo folder containing data processing scripts/notebooks&#41;)

[comment]: <> (4. etc...)

[comment]: <> (*If your project is well underway and setup is fairly complicated &#40;ie. requires installation of many packages&#41; create another "setup.md" file and link to it here*  )

[comment]: <> (5. Follow setup [instructions]&#40;Link to file&#41;)

[comment]: <> (## Featured Notebooks/Analysis/Deliverables)

[comment]: <> (* [Notebook/Markdown/Slide Deck Title]&#40;link&#41;)

[comment]: <> (* [Notebook/Markdown/Slide DeckTitle]&#40;link&#41;)

[comment]: <> (* [Blog Post]&#40;link&#41;)
