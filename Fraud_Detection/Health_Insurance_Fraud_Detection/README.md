# Health Insurance Fraud Detection

Description: Developed and deployed machine learning models to detect fraudulent activities in health insurance claims. The project involved extensive data preprocessing, model selection, hyperparameter tuning, and performance evaluation to identify patterns indicative of fraud.

Rigorous analysis of Medicare data has yielded many physicians who indulge in fraud. They adopt ways in which an ambiguous diagnosis code is used to adopt the costliest procedures and drugs. Insurance companies are the most vulnerable institutions impacted due to these bad practices. Due to this, insurance companies increased their insurance premiums, and as a result, healthcare is becoming costly daily.

Healthcare fraud and abuse take many forms. Some of the most common types of fraud by providers are:

a) Billing for services that were not provided.

b) Duplicate submission of a claim for the same service.

c) Misrepresenting the service provided.

d) Charging for a more complex or expensive service than was actually provided.

e) Billing for a covered service when the service actually provided was not covered.

Problem Statement
This project aims to " predict the potentially fraudulent providers " based on the claims filed by them. We will also discover important variables that help detect the behavior of potential fraud providers. Further, we will study fraudulent patterns in the provider's claims to understand the future behavior of providers.

## Pipeline File
[Health Insurance Fraud Detection Model](./Health_Insurance_Fraud_Detection_Pipeline.py)

## Libraries and Frameworks:

- **scikit-learn**: Implemented machine learning models, model evaluation metrics, hyperparameter tuning, and data preprocessing.
- **SciPy**: Utilized indirectly through scikit-learn for optimization and statistical functions.
- **XGBoost**: Used for gradient boosting models.

## Machine Learning Algorithms:

- **Logistic Regression**
- Support Vector Machines (SVM) with **SVC** and **LinearSVC**
- Naive Bayes (**GaussianNB**)
- **Decision Trees**
- **Random Forest**
- XGBoost (**XGBClassifier**)

## Model Evaluation and Tuning:

- **GridSearchCV** and **RandomizedSearchCV** for hyperparameter tuning
- Confusion Matrix, ROC Curve, AUC, F1 Score for model evaluation
- **CalibratedClassifierCV** for probability calibration of classifiers
- Classification Report generation for detailed performance metrics

## Data Preprocessing:
- Data splitting with **train_test_split**
- Feature scaling using **StandardScaler**

## Data Visualization:

- Used **Seaborn** and **Matplotlib** for data visualization and plotting of model performance metrics.

## Programming and Scripting:

- **Python** programming in a Jupyter Notebook environment
- Data manipulation with **NumPy** and **Pandas**
- File and directory handling using **os** module

## Project Outcomes:

Achieved a significant improvement in fraud detection accuracy, resulting in more accurate identification of fraudulent claims.
The best-performing model achieved an AUC (Area Under the Curve) score of 0.94, demonstrating strong discriminatory power between fraudulent and non-fraudulent claims.
Reduced false positives by optimizing model thresholds, improving the precision of fraud detection while minimizing disruptions to legitimate claims.
Developed a scalable and reusable pipeline for fraud detection that can be adapted for similar datasets in the future.
Visualized key insights and model performance metrics, aiding stakeholders in understanding model decisions and overall fraud detection strategy.

## Algorithm Versions 
### Health_Insurance_Fraud_Detection_1
This was my first exposure to the dataset. The data-wrangling mistake was removing data points instead of drawing inferences from data such as DOB, time admitted, time discharged, etc. The trained models poorly detected fraudulent transactions.

### Health_Insurance_Fraud_Detection_2
Second run into data wrangling, I added more data (columns) to the dataset to provide numerical, adequate data for the models to be better trained. After utilizing some data to create more data numerically. I removed the 'no-longer-needed' data. The trained models significantly improved. However, I believe the models can improve if I balance the data.

### Health_Insurance_Fraud_Detection_3
The data wrangling technique stayed the same but I then removed some of the oversampled data 'Non Fraudulent' data points to match the count with 'Fraudulent' data points. The trained models significantly improved in catching fraudulent transactions; however, the ROC-AUC score decreased due to the low data volume resulting from undersampling. I believe the model can be improved by utilizing alternative methods to combat imbalanced data without undersampling and eliminating too many data points.

### Health_Insurance_Fraud_Detection_4
The data wrangling technique stayed the same, but I used the SMOTE oversampling technique to create emulated undersampled data points to equalize the dataset. The trained models did better at detecting fraudulent transactions; The highest AUC scores were reported with this method on training the models. However, it called too many Non-Fraudulent transactions Fraudulent. This requires further fitting of the SMOTE resampling technique to optimize the emulated data to train the models appropriately.

## Dataset
Dataset: [Healthcare_Provider_Fraud_Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data)

## Scientific Paper(s) Utilized
[Statistical Methods for Health Care Fraud Detection](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/4/216/files/2015/09/p70-Statistical-Methods-for-Health-Care-Fraud-Detection.pdf)
