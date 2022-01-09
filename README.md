# Tennis Match Outcome Prediction Model

This is an individual project created through the Project Track of
the [Snowball Initiative](https://dataclub.northeastern.edu/snowball/)
at [Northeastern Data Club](http://www.https://dataclub.northeastern.edu/).

#### -- Project Status: Active

## Project Description

* Created a tool that predicts tennis match outcomes (mean accuracy ~ 0.8276) to help players, coaches, and fans to
  better understand factors that may influence win likelihood
* Engineered features from the play-by-play records of first sets to quantify player performance
* Performed recursive feature selection to separate most relevant features
* Optimized Logistic Regression, kNN, Decision Trees, Naive Bayes, Linear SVM using GridsearchCV to reach the best
  model.
* Built a client facing API using streamlit

## Purpose/Objective

The purpose of this project is to build a Classification Machine Learning model that can predict the outcome (winner) of
a tennis match, given the play-by-play data of the first set. The target applications of this model are widespread: it
can be leveraged as an informative resource for sports-betting, a guide to players and coaches on the improvements that
will maximize winning potential, and much more. While the model is currently trained on data from professional men's
tennis matches, it can be expanded in the future for compatibility with the women's tour and even for
recreational/casual players.

## Code and Resources Used

**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, streamlit, pickle  

[comment]: <> (**For Web Framework Requirements:**  ```pip install -r requirements.txt```  )
**Dataset GitHub:** https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv  
**Streamlit Productionization:** https://github.com/dataprofessor/code/tree/master/streamlit/part7

## Methods Used

* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling

## Data Collection and Cleaning

I acquired match data
from [Jeff Sackmann's Github](https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv)
.
\
\
After downloading the data, I needed to clean it up so that it was usable for the model. I made the following changes
and created the following variables:

* Removed matches from Wimbledon's Final Round Qualifying (they have different rules)
* Parsed the 'play-by-play' strings into useful statistics with custom functions that extract relevant data (points,
  aces, breaks)
* Feature engineered momentum to quantify trends not explicit in the data (based on consecutive points won)
* Made a new column for points scored for each player
* Made a new column for momentum accumulated for each player
* Made a new column for breaks won for each player
* Made a new column for aces served for each player
* Resulted with 925 samples and 23 features

## EDA

I looked at the distributions of the data and the value counts for the various quantitative and categorical variables.
Below are a few highlights from my analysis.

![alt text](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/blob/main/visualizations/s1_s2_points_win_relplot.png "First Set Points Colored By Winner")
![alt text](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/blob/main/visualizations/correlation_heatmap.png "Correlations")
![alt text](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/blob/main/visualizations/s1_s2_points_histogram.png "First Set Points Distribution")

## Model Building

First, I split the data into train and tests sets with a test size of 25%.

I tried five different models and evaluated them using mean accuracy. I chose mean accuracy because it is relatively
easy to interpret and the data is fairly balanced so mean accuracy provides an accurate measure for model performance.

I also performed recursive feature elimination to select the most relevant features

Finally, I used GridSearchCV to perform hyperparameters tuning

I used five different models:

* **Logistic Regression** – Highly efficient to train and classify new instances, easy to interpret.
* **kNN Classifier** – Intuitive algorithm and makes no assumptions about the data and its distribution.
* **Support Vector Machine** – Effective for classification in high dimensional spaces.
* **Gaussian Naive Bayes** – Similar to Logistic Regression, highly efficient to train and classify new instances, easy
  to interpret.
* **Decision Tree** – Similar to kNN Classifier, intuitive algorithm and makes no assumptions about the data and its
  distribution.

## Model performance

After tuning, all of the models had equal performance on the test and validation sets.

* **Logistic Regression**: Accuracy = 0.8276
* **kNN Classifier**: Accuracy = 0.8276
* **Support Vector Machine**: Accuracy = 0.8276
* **Gaussian Naive Bayes**: Accuracy = 0.8276
* **Decision Tree**: Accuracy = 0.8276

This led me to believe that one of the features was heavily influencing the outcome of the models, and I discovered that
the first set winner feature was directly being used to predict the match winner. Thus, none of the other features were 
creating additional insights.

## Productionization

In this step, I built a Streamlit API that was hosted on a local webserver by following along with the tutorial in the
reference section above. The API takes in a first set play-by-play encoded string and computes match statistics and
returns a prediction for the winner of the match.

## Project Structure

- raw dataset and preprocessed dataset are included under
  the [data](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/data) directory
- model and scaler objects are included under
  the [models](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/models) directory
- Jupyter Notebook work is included under
  the [notebooks](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/notebooks) directory
- all source code (data wrangling, exploratory data analysis, model building, custom functions) is included under
  the [src](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/src) directory
- produced visualizations are included under
  the [visualizations](https://github.com/jjz17/Tennis-Match-Outcome-Prediction/tree/main/visualizations) directory
