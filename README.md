# Analyzing Brain Stroke Data: An Exploration of Patient Attributes and Risk Factors

# Team Members:
 Lauren Carter,
 Domenic Guerrero,
 Sandeep Singh,
 Kevin Curry,
 Micah Purdome

## Overview
The primary focus of this project is to compare and contrast the nature and performance of five separate machine learning models. We specifically focus on the causal relationships related to brain strokes in America, particularly concerning age, glucose levels, and BMI, and how these factors affect lethality. 

##Links to data
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
   - **Logistic Regression**: Applied with both a simple train-test split and stratified cross-validation, Logistic Regression is used as a baseline classifier. The stratified approach ensures that the distribution of target classes (stroke vs. no stroke) is proportionally represented in each training and validation split.
   - **K-Nearest Neighbors (KNN)**: This algorithm is chosen for its ability to capture local patterns in the data. Similar to Logistic Regression, it is tested with stratified cross-validation to provide a balanced view of performance.
   - **Random Forest Classifier**: As an ensemble method, Random Forests are particularly useful in handling complex, high-dimensional data. The project includes experiments with cross-validation to assess the model's robustness.
   - **Gradient Boosting Classifier**: Known for its strong predictive performance, Gradient Boosting is also assessed under cross-validation. This classifier adds weak learners iteratively to correct previous errors, making it highly adaptive.
   - **Decision Tree Classifier**: Decision Trees are employed for their interpretability and ease of use. Their performance is evaluated using cross-validation to mitigate the risk of overfitting.
### 3. **Model Evaluation and Hyperparameter Tuning**
   - Each model undergoes **Grid Search** for hyperparameter tuning, where various parameter combinations are systematically tested to identify the optimal configuration for each model.
   - **Stratified Cross-Validation** is consistently applied, ensuring that the evaluation metrics represent the model’s true performance across all target classes.
