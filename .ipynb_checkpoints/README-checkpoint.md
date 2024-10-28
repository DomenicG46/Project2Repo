# Project2Repo

# Team Members:
 Lauren Carter
 Domenic Guerrero
 Sandeep Singh
 Kevin Curry
 Micah Pardome

# BrainStroke Analysis and Brainstroke Prediction Model

## Overview




## Data Sources

The project uses the following datasets which are stored in the Resource folder:

1. **Big Cities Health**: BigCitiesHealth.csv
2. **Brain Stroke Data**: "brain_stroke.csv"


Links to data
- https://bigcitieshealthdata.org/
- https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset
- https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset

## Requirements

- Python 3.8+
- The following Python libraries are required:
  - pandas
  - numpy
  - matplotlib
  SkLearn Libraries
  - train_test_split, StratifiedKFold
  - StandardScaler, LabelEncoder
  - accuracy_score, classification_report, confusion_matrix
  - LogisticRegression
  - KNeighborsClassifier
  - RandomForestClassifier
  - GradientBoostingClassifier
  - DecisionTreeClassifier
  - preprocessing
  from statistics import mean, stdev
  from xgboost import XGBClassifier
    
## Installation

1. Clone the repository:
   https://github.com/DomenicG46/Project2Repo.git
   
## Usage
## 1. Loading the Data
Load the required datasets using pandas:

import pandas as pd
from pathlib import Path

## Load datasets
   ??Projects/Repositories/Repo's/Project2Repo/brain_stroke.csv

## 2. Data Cleaning and Preparation
-   In the BigCitiesHealth.csv we filtered through dataset to find the two prediction variables. There was alot
  of data and we we went through thoroughly. After we found some interesting variables we focused on one particular cause of death.
  Brainstrokes were an interesting cause of death and we searched for Data Set.
- We then found Brainstroke.csv and we searched from there. We then cleaned the data to simplify our prediction variiables.
- 
## 3. Data Visualization
The project uses Matplotlib to generate various plots and StandardScaler to set the x & y paramaeters.

## Running Prediction Models
The following prediction models were then used to find the best prediction method:
- LogisticRegression (Lauren Carter?) - Maximum Accuracy That can be obtained from this model is: 95.18072289156626 %
                                       Overall Accuracy: 95.02108634940564 %
- KNeighbors (Domenic Guerrero) - Maximum Accuracy That can be obtained from this model is: 95.18072289156626 %
                                  Overall Accuracy: 94.74008257478813 %
- RandomForestClassifier (?)
- 
# Key Features of Project?:
- Write Here
- another feature

## Files Attached
- 
- 
- 
- 
## Overall Project Analysis/Results
T


## License




## Resources




