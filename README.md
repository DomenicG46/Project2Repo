# Analyzing Brain Stroke Data: An Exploration of Patient Attributes and Risk Factors

# Team Members:
 Lauren Carter,
 Domenic Guerrero,
 Sandeep Singh,
 Kevin Curry,
 Micah Purdome

## Overview
The primary focus of this project is to compare and contrast the nature and performance of five separate machine learning models. We specifically focus on the causal relationships related to brain strokes in America, particularly concerning age, glucose levels, and BMI, and how these factors affect lethality. 

## Links to data
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


### 1. **Data Preprocessing**
   - **Checking for Missing Values**: The dataset is examined for any missing or null values to ensure data integrity. Handling missing data is critical as it can lead to biased predictions or reduce the model's effectiveness.
   - **Attribute Data Types**: The data types of each attribute are verified to ensure compatibility with machine learning models. Proper data type allocation helps prevent errors during model training and can enhance computational efficiency.
### 2. **Model Selection and Implementation**
   The project utilizes multiple machine learning classifiers, and each model undergoes specific testing techniques to ensure it generalizes well to unseen data:
   - **Logistic Regression**:
   - **K-Nearest Neighbors (KNN)**:
   - **Random Forest Classifier**:
   - **Gradient Boosting Classifier**: 
   - **Decision Tree Classifier**:
### 3. **Model Evaluation and Hyperparameter Tuning**
   - Each model undergoes **Grid Search** for hyperparameter tuning, where various parameter combinations are systematically tested to identify the optimal configuration for each model.
   - **Stratified Cross-Validation** 


## Analysis: 

The models that were used, like Random Forest, K Nearest Neighbors, Logistic Regression, and Gradient Boost, all got a high accuracy of 0.95. The Decision Tree was still good but a little lower, at 0.91.The utilization of grid search is optimal only in the case of the decision tree, where its implementation resulted in a 4% increase in performance.Our analysis concluded that the random forest was the most optimal model.
