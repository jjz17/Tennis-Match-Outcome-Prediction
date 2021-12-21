# Tennis Match Outcome Prediction Model
This project is an individual project created through the Project Track of the [Snowball Initiative](https://dataclub.northeastern.edu/snowball/) at [Northeastern Data Club](http://www.https://dataclub.northeastern.edu/).

#### -- Project Status: Active    [Active, On-Hold, Completed]

## Project Intro/Objective
The purpose of this project is to build a Classification Machine Learning model that can predict the outcome (winner) of a tennis match, given the play-by-play data of the first set. The target applications of this model are widespread: it can be leveraged as an informative resource for sports-betting, a guide to players and coaches on the improvements that will maximize winning potential, and much more. While the model is currently trained on data from professional men's tennis matches, it can be expanded in the future for compatibility with the women's tour and even for recreational/casual players.

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* Pandas
* Jupyter
* Scikit-Learn
* Matplotlib

## Project Description
### The Problem
The outcomes of tennis matches are notoriously difficult to predict, due to the volatile nature of the sport: changes in momentum, effects from the audience, and a variety of other factors all contribute to its unpredictability. In this project, I tackle this historical challenge, by building a Machine Learning model that predicts the outcome of tennis matches solely based upon play-by-play data from the first set. 
### The Data
I acquired match data from [Jeff Sackmann's Github](https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv). To parse the 'play-by-play' strings into useful statistics, I wrote a family of functions that extract relevant data (points, aces, breaks). Furthermore, I performed feature engineering by quantifying the 'momentum' of a player based on consecutive points won. Through the process of data acquisition, cleaning, and wrangling, I ended up with 925 samples and 23 features. 
\
\
To do this, I plan on studying and normalizing tennis match data to discover particular features, and to engineer features of my own, which will be fed into a machine learning algorithm to have the match outcomes predicted. I will be looking to discover which variables are most indicative of the match outcomes, and to attempt to create new variables from the existing ones which will further aid the machine learning models. I also plan on testing the data with a variety of different models to determine which one produces the most accurate results.
\
\
I will be testing a variety of classification algorithms, including Logistic Regression, k-Nearest Neighbors, etc.
\
\
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modeling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Needs of this project

- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- etc. (be as specific as possible)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)


## Contributing DSWG Members

**Team Leads (Contacts) : [Full Name](https://github.com/[github handle])(@slackHandle)**

#### Other Members:

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |

## Contact
* If you haven't joined the SF Brigade Slack, [you can do that here](http://c4sf.me/slack).  
* Our slack channel is `#datasci-projectname`
* Feel free to contact team leads with any questions or if you are interested in contributing!
