# Project
Credit Card Approval Prediction

This repository contains a comprehensive machine learning pipeline for predicting the credit risk of applicants based on their personal and financial data. The dataset includes various demographic and financial features, and the goal is to classify applicants as either good or bad credit risks. The code covers data preprocessing, feature engineering, exploratory data analysis (EDA), and building machine learning models to predict credit risk.

Files

application_record.csv - The main dataset containing applicant information.
credit_record.csv - Contains the credit status of the applicants across multiple months.
Project.ipynb - Colab notebook version of the pipeline for interactive use.

Libraries Used

pandas - Data manipulation and analysis.

numpy - Mathematical operations.

matplotlib - Data visualization.

seaborn - Statistical data visualization.

scikit-learn - Machine learning algorithms and tools.

imbalanced-learn (imblearn) - Handling imbalanced datasets with techniques like SMOTE.

RandomizedSearchCV and GridSearchCV - Hyperparameter tuning.

LabelEncoder - Label encoding for categorical variables.

Workflow

Data Loading & Merging
The application and credit record datasets are loaded and merged based on the applicant ID.

Data Cleaning & Feature Engineering
Missing values are handled, categorical variables are encoded, and new features are created:

APPLICANT_AGE is calculated from DAYS_BIRTH.

YEARS_EMPLOYED is derived from DAYS_EMPLOYED.

Family status and housing type are standardized.

Income groups and occupation types are assigned.

Exploratory Data Analysis (EDA)
Gender distribution, delinquency rates based on marital status, family size, and income levels are analyzed.
Various plots (e.g., pie chart, count plot, scatter plot) are created to understand trends and relationships in the data.

Outlier Detection and Removal
Outliers are detected and removed using the Interquartile Range (IQR) method.

Label Encoding
Categorical features are encoded into numerical values using Label Encoding.

Model Building
Several machine learning models are trained to predict credit risk:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Classifier (SVC)
RandomizedSearchCV and GridSearchCV are used for hyperparameter tuning.

Evaluation
The models are evaluated using metrics like accuracy score, classification report, confusion matrix, ROC curve, and AUC score.

SMOTE for Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance by oversampling the minority class in the training set.

Model Evaluation
After training the models, they are evaluated on the test set, and the following results are printed for each model:

Accuracy Score

Classification Report

Confusion Matrix

ROC Curve and AUC Score

Hyperparameter Tuning

Hyperparameters are tuned using RandomizedSearchCV for Logistic Regression, GridSearchCV for Linear SVC, Decision Tree, and Random Forest models to improve model performance.

Feature Importance
Feature importance is computed for models like Random Forest, Decision Tree, Logistic Regression, and SVC, and visualized for model interpretability.

How to Run the Code

Requirements
Make sure you have the following libraries installed:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

Steps

Place the dataset files application_record.csv and credit_record.csv in the project directory or update the file paths in the code accordingly.

open the Colab notebook:

Colab notebook Project.ipynb

You will get an output with model evaluation results, including accuracy, ROC curves, confusion matrix plots, and feature importance plots.

Example Output

Gender Distribution: A pie chart showing the gender distribution of applicants.
Delinquency Rate by Gender/Family Status: A count plot illustrating the delinquency rates based on gender and family status.

Income vs Employment: A scatter plot showing the relationship between total income and years of employment.

Model Performance: A classification report and confusion matrix displaying how well the models predict the credit risk.

Feature Importance: Visualizations of feature importances for models like Random Forest, Decision Tree, Logistic Regression, and SVC.

Conclusion

This project demonstrates a full data analysis and machine learning pipeline for predicting credit risk. It uses common techniques like data preprocessing, feature engineering, model selection, hyperparameter tuning, and handling class imbalance. You can use the same methodology to process and predict credit risks on other datasets by adjusting feature engineering steps or models based on the problem at hand.

Contributing

Contributions to improve the model, analysis, or documentation are welcome. If you have any suggestions or improvements, feel free to open an issue or submit a pull request.
