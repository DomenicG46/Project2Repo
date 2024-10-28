# Project2Repo


# BrainStroke Analysis and Brainstroke Prediction Model

# Team Members:
 Lauren Carter
 Domenic Guerrero
 Sandeep Singh
 Kevin Curry
 Micah Pardome


## Overview
The primary focus of this project is to compare and contrast the nature and performance of five separate machine learning models. We specifically focus on the causal relationships related to brain strokes in America, particularly concerning age, glucose levels, and BMI, and how these factors affect lethality. 

Links to data
- https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-datase

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


**Data Preprocessing**: Checking for missing values and assessing data types.
2. **Model Selection and Evaluation**:
   - Several classifiers are used, including Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, Gradient Boosting, and Decision Trees.
   - Each classifier is evaluated using stratified cross-validation to ensure balanced class representation.
3. **Hyperparameter Tuning**:
   - Grid search is employed to optimize hyperparameters for each model, enhancing prediction accuracy.
