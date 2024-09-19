import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc , f1_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def data_wrangle(folder):
    files = []
    for file in os.listdir("Dataset_Health_Insurance"):
        file_path = os.path.join("Dataset_Health_Insurance", file)
        files.append(file_path)

    claims_code = pd.read_csv(files[0])
    df_test = pd.read_csv(files[1])
    df_test_beneficiary_data = pd.read_csv(files[2])
    df_train_inpatient_data = pd.read_csv(files[3])
    df_train_outpatient_data = pd.read_csv(files[4])
    df_train = pd.read_csv(files[5])
    df_train_beneficiary_data = pd.read_csv(files[6])
    df_train_inpatient_data = pd.read_csv(files[7])
    df_train_outpatient_data = pd.read_csv(files[8])

    # replace 2 to 0 for the chronic conditions to indicate False.
    df_train_beneficiary_data = df_train_beneficiary_data.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                               'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                               'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                               'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)
    # For RenalDiseaseIndicator replacing 'Y' with 1
    df_train_beneficiary_data = df_train_beneficiary_data.replace({'RenalDiseaseIndicator': 'Y'}, 1)

    # convert all these columns datatypes to numeric
    df_train_beneficiary_data[["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease", "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", 
                               "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke", 
                               "RenalDiseaseIndicator"]] =df_train_beneficiary_data[["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease", 
                                                                                    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", 
                                                                                    "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke", "RenalDiseaseIndicator"]].apply(pd.to_numeric)
    
    df_train_beneficiary_data['Patient_Risk_Score'] = df_train_beneficiary_data['ChronicCond_Alzheimer'] +df_train_beneficiary_data['ChronicCond_Heartfailure'] + \
                                            df_train_beneficiary_data['ChronicCond_KidneyDisease'] +df_train_beneficiary_data['ChronicCond_Cancer'] +\
                                            df_train_beneficiary_data['ChronicCond_ObstrPulmonary'] + df_train_beneficiary_data['ChronicCond_Depression'] +\
                                            df_train_beneficiary_data['ChronicCond_Diabetes'] + df_train_beneficiary_data['ChronicCond_IschemicHeart'] +\
                                            df_train_beneficiary_data['ChronicCond_Osteoporasis'] +df_train_beneficiary_data['ChronicCond_rheumatoidarthritis'] +\
                                            df_train_beneficiary_data['ChronicCond_stroke'] +df_train_beneficiary_data['RenalDiseaseIndicator']
    
    # Replacing '2' with '0' for Gender Type
    df_train_beneficiary_data = df_train_beneficiary_data.replace({'Gender': 2}, 0)

    # Convert Date of Birth and Date of Death from String to Datetime format
    df_train_beneficiary_data['DOB'] = pd.to_datetime(df_train_beneficiary_data['DOB'] , format = '%Y-%m-%d')
    df_train_beneficiary_data['DOD'] = pd.to_datetime(df_train_beneficiary_data['DOD'],format = '%Y-%m-%d')

    # Get the birth month and Birth year for DOB and DOD
    df_train_beneficiary_data['Birth_Year'] = df_train_beneficiary_data['DOB'].dt.year
    df_train_beneficiary_data['Birth_Month'] = df_train_beneficiary_data['DOB'].dt.month

    df_train_beneficiary_data['Patient_Age'] = round(((df_train_beneficiary_data['DOD'] - df_train_beneficiary_data['DOB']).dt.days)/365)
    df_train_beneficiary_data.Patient_Age.fillna(round(((pd.to_datetime('2009-12-01',format ='%Y-%m-%d')-df_train_beneficiary_data['DOB']).dt.days)/365),inplace=True)

    # Set value=1 if the patient is dead i.e DOD value is not null
    df_train_beneficiary_data['isDead'] = 0
    df_train_beneficiary_data.loc[df_train_beneficiary_data.DOD.notna(), 'isDead'] = 1

    # convert ClaimStartDt, ClaimEndDt from string to datetime format
    df_train_inpatient_data['ClaimStartDt'] = pd.to_datetime(df_train_inpatient_data['ClaimStartDt'] , format = '%Y-%m-%d')
    df_train_inpatient_data['ClaimEndDt'] = pd.to_datetime(df_train_inpatient_data['ClaimEndDt'],format = '%Y-%m-%d')

    # convert AdmissionDt, DischargeDt from string to datetime format
    df_train_inpatient_data['AdmissionDt'] = pd.to_datetime(df_train_inpatient_data['AdmissionDt'] , format = '%Y-%m-%d')
    df_train_inpatient_data['DischargeDt'] = pd.to_datetime(df_train_inpatient_data['DischargeDt'],format = '%Y-%m-%d')

    # Calculate Hospitalization_Duration = DischargeDt - AdmissionDt
    df_train_inpatient_data['Hospitalization_Duration'] = ((df_train_inpatient_data['DischargeDt'] - df_train_inpatient_data['AdmissionDt']).dt.days)+1
    # Calculate Claim_Period = ClaimEndDt - ClaimStartDt
    df_train_inpatient_data['Claim_Period'] = ((df_train_inpatient_data['ClaimEndDt'] - df_train_inpatient_data['ClaimStartDt']).dt.days)+1

    # ExtraClaimDays = Claim_Period - Hospitalization_Duration
    df_train_inpatient_data['ExtraClaimDays'] = np.where(df_train_inpatient_data['Claim_Period']>df_train_inpatient_data['Hospitalization_Duration'], df_train_inpatient_data['Claim_Period'] - df_train_inpatient_data['Hospitalization_Duration'], 0)

    # Get the months and year of claim start and claim end
    df_train_inpatient_data['ClaimStart_Year'] = df_train_inpatient_data['ClaimStartDt'].dt.year
    df_train_inpatient_data['ClaimStart_Month'] = df_train_inpatient_data['ClaimStartDt'].dt.month
    df_train_inpatient_data['ClaimEnd_Year'] = df_train_inpatient_data['ClaimEndDt'].dt.year
    df_train_inpatient_data['ClaimEnd_Month'] = df_train_inpatient_data['ClaimEndDt'].dt.month
   
    # Get the month and year of Admission_Year and Admission_Month
    df_train_inpatient_data['Admission_Year'] = df_train_inpatient_data['AdmissionDt'].dt.year
    df_train_inpatient_data['Admission_Month'] = df_train_inpatient_data['AdmissionDt'].dt.month
    df_train_inpatient_data['Discharge_Year'] = df_train_inpatient_data['DischargeDt'].dt.year
    df_train_inpatient_data['Discharge_Month'] = df_train_inpatient_data['DischargeDt'].dt.month
    
    # convert ClaimStartDt, ClaimEndDt from string to datetime format
    df_train_outpatient_data['ClaimStartDt'] = pd.to_datetime(df_train_outpatient_data['ClaimStartDt'] , format = '%Y-%m-%d')
    df_train_outpatient_data['ClaimEndDt'] = pd.to_datetime(df_train_outpatient_data['ClaimEndDt'],format = '%Y-%m-%d')

    # Get the months and year of claim start and claim end
    df_train_outpatient_data['ClaimStart_Year'] = df_train_outpatient_data['ClaimStartDt'].dt.year
    df_train_outpatient_data['ClaimStart_Month'] = df_train_outpatient_data['ClaimStartDt'].dt.month
    df_train_outpatient_data['ClaimEnd_Year'] = df_train_outpatient_data['ClaimEndDt'].dt.year
    df_train_outpatient_data['ClaimEnd_Month'] = df_train_outpatient_data['ClaimEndDt'].dt.month

    # Calculate Claim_Period = ClaimEndDt - ClaimStartDt
    df_train_outpatient_data['Claim_Period'] = ((df_train_outpatient_data['ClaimEndDt'] - df_train_outpatient_data['ClaimStartDt']).dt.days)+1

    # Create a new column Inpatient_or_Outpatient where Inpatient =1 and Outpatient = 0
    df_train_inpatient_data['Inpatient_or_Outpatient'] = 1
    df_train_outpatient_data['Inpatient_or_Outpatient'] = 0

    # Merge inpatient and outpatient dataframes based on common columns
    common_columns_test = [idx for idx in df_train_outpatient_data.columns if idx in df_train_inpatient_data.columns]
    df_in_out_patient_test = pd.merge(df_train_inpatient_data, df_train_outpatient_data, left_on = common_columns_test, right_on = common_columns_test,how = 'outer')

    # Merge beneficiary details with inpatient and outpatient data
    df_in_out_patient_test = pd.merge(df_in_out_patient_test, df_train_beneficiary_data, left_on='BeneID',right_on='BeneID',how='inner')

    df_master = df_in_out_patient_test.merge(df_train[['Provider', 'PotentialFraud']], on='Provider', how='left')

    # create new feature total reimbursement amount for inpatient and outpatient
    df_master['IP_OP_TotalReimbursementAmt'] = df_master['IPAnnualReimbursementAmt'] + df_master['OPAnnualReimbursementAmt']
    # create new feature total deductible amount for inpatient and outpatient
    df_master['IP_OP_AnnualDeductibleAmt'] = df_master['IPAnnualDeductibleAmt'] + df_master['OPAnnualDeductibleAmt']

    # Fill missing results using 0
    df_master = df_master.fillna(0).copy()

    def create_feature_using_groupby(df, groupby_cols, operation_cols, operation):
        '''
        This function groupby the dataframe by 'groupby_cols' and performs 'operation' on 'operation_cols'.
        '''
        for col in operation_cols:
            for groupby_col in groupby_cols:
                # create new column name for the dataframe
                new_col_name = f'Per{"_".join(groupby_col)}_{operation}_{col}'
                df[new_col_name] = df.groupby(groupby_col)[col].transform(operation)
        return df
    
    # Group by columns and columns to apply the operation
    groupby_columns = [
        ['Provider'], ['BeneID'], ['AttendingPhysician'], ['OperatingPhysician'],
        ['OtherPhysician'], ['DiagnosisGroupCode'], ['ClmAdmitDiagnosisCode'],
        ['ClmProcedureCode_1'], ['ClmProcedureCode_2'], ['ClmProcedureCode_3'],
        ['ClmProcedureCode_4'], ['ClmProcedureCode_5'], ['ClmProcedureCode_6'],
        ['ClmDiagnosisCode_1'], ['ClmDiagnosisCode_2'], ['ClmDiagnosisCode_3'],
        ['ClmDiagnosisCode_4'], ['ClmDiagnosisCode_5'], ['ClmDiagnosisCode_6']
    ]

    operation_columns = [
        'InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt',
        'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
        'Patient_Age', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
        'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score'
    ]

    # Apply the transformation
    df_master = create_feature_using_groupby(df_master, groupby_columns, operation_columns, 'mean')

    
    # count the claims per provider
    df_master =  create_feature_using_groupby(df_master, ['Provider'], ['ClaimID'], 'count')

    
    columns = ['ClaimID']
    grp_by_cols = ['BeneID', 'AttendingPhysician', 'OtherPhysician', 'OperatingPhysician', 'ClmAdmitDiagnosisCode', 'ClmProcedureCode_1',
                   'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
                   'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'DiagnosisGroupCode']
    for ele in grp_by_cols:
        lst = ['Provider', ele]
        df_master =  create_feature_using_groupby(df_master, lst, columns, 'count')

    # remove the columns which are not required
    remove_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician','OperatingPhysician', 'OtherPhysician',
                    'ClmDiagnosisCode_1','ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4','ClmDiagnosisCode_5',
                    'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7','ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
                    'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3','ClmProcedureCode_4', 'ClmProcedureCode_5',
                    'ClmProcedureCode_6','ClmAdmitDiagnosisCode', 'AdmissionDt','ClaimStart_Year', 'ClaimStart_Year', 'ClaimStart_Month',
                    'ClaimEnd_Year', 'ClaimEnd_Month', 'Admission_Year', 'Admission_Month', 'Discharge_Year', 'Discharge_Month',
                    'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD','Birth_Year', 'Birth_Month','State', 'County']

    df_master2 = df_master.drop(columns=remove_columns, axis=1)

    # Convert type of Gender and Race to categorical
    df_master2.Gender=df_master2.Gender.astype('category')
    df_master2.Race=df_master2.Race.astype('category')
    # Do one hot encoding for gender and Race
    df_master2=pd.get_dummies(df_master2,columns=['Gender','Race'])

    def process_fraud_data(df_master2, df_train):
        if "PotentialFraud" in list(df_train.columns):
            df_final = df_master2.groupby(['Provider', 'PotentialFraud'], as_index=False).agg('sum')
            df_final.PotentialFraud.replace(['Yes', 'No'], ['1', '0'], inplace=True)
            df_final.PotentialFraud = df_final.PotentialFraud.astype('int64')
            return df_final
        else:
            df_final = df_master2.groupby(['Provider'], as_index=False).agg('sum')
            return df_final

    df_final = process_fraud_data(df_master2, df_train)

    # Move 'PotentialFraud' column to the end
    column_order = [col for col in df_final.columns if col != 'PotentialFraud'] + ['PotentialFraud']
    df_final = df_final[column_order]

    # Balance the data Fraud/Not-Fraud classifications
    df_final2 = df_final.copy()
    # check the number of instances of 'Frauds' and 'Non-Frauds' to check class imbalance.
    print("Instances before Balance: ",df_final2['PotentialFraud'].value_counts())
    # shuffle the balanced dataset to ensure randomness after concatenation.
    df_final2_b = df_final2.sample(frac=1)

    # Total of 5410 transactions. 
    # Split the dataset into training, testing, and validation sets
    # Training Set: 70% (3787)
    x_train = df_final2.iloc[:3787, :-1]
    y_train = df_final2.iloc[:3787, -1].astype(int)

    # Test Set: 15% (811)
    x_test = df_final2.iloc[3787:4598, :-1]
    y_test = df_final2.iloc[3787:4598, -1].astype(int)

    # Validation Set: 15% (812)
    x_val = df_final2.iloc[4598:, :-1]
    y_val = df_final2.iloc[4598:, -1].astype(int)

    #hold the provider values if needed
    x_train_P = x_train[['Provider']]
    x_test_P = x_test[['Provider']]
    x_val_P = x_val[['Provider']]

    x_train = x_train.drop(columns=['Provider'], axis=1)
    x_test = x_test.drop(columns=['Provider'], axis=1)
    x_val = x_val.drop(columns=['Provider'], axis=1)

    return x_train, y_train, x_test, y_test, x_val, y_val

def standardization_and_sample_balance(x_train, x_val, x_test, y_train):  
    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)
    x_train = standard_scaler.transform(x_train)

    standard_scaler = StandardScaler()
    standard_scaler.fit(x_val)
    x_val = standard_scaler.transform(x_val)

    standard_scaler = StandardScaler()
    standard_scaler.fit(x_test)
    x_test = standard_scaler.transform(x_test)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, x_val, x_test, y_train

def ensemble_model(x_train, y_train, x_test, y_test, x_val, y_val):
    #1. Train the models
    ## XGBoost Model
    def xgboost_model(x_train, x_val, x_test, y_train, y_val, y_test):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        dtest = xgb.DMatrix(x_test, label=y_test)

        # Set up XGBoost parameters
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.3,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'seed': 42
        }

        watchlist = [(dtrain, 'train'), (dval, 'eval')]  # Track training and validation performance

        model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, early_stopping_rounds=10)

        y_pred_prob1 = model.predict(dval)  # Predicted probabilities
        y_pred1 = [1 if y > 0.5 else 0 for y in y_pred_prob1]  # Convert probabilities to binary predictions

        return y_pred_prob1, y_pred1
    
    y_pred_prob1, y_pred1 = xgboost_model(x_train, x_val, x_test, y_train, y_val, y_test)

    ## Logistic Regression Model
    def log_reg_model(x_train, y_train, x_val):
        logistic_model = LogisticRegression()
        logistic_model.fit(x_train, y_train)
        y_val_pred_prob2 = logistic_model.predict_proba(x_val)[:, 1] # Predicted probabilities for class 1 (Fraudulent)
        y_pred2 = logistic_model.predict(x_val)
        return y_val_pred_prob2, y_pred2
    
    y_pred_prob2, y_pred2 = log_reg_model(x_train, y_train, x_val)

    ## Linear SVC Model
    # Train Linear SVC
    def linear_svc_model(x_train, y_train, x_val):
        svc = LinearSVC(class_weight='balanced')
        svc.fit(x_train, y_train)
        # Predict class labels
        y_pred3 = svc.predict(x_val)
        # Get decision function scores for ROC-AUC
        y_pred_prob3 = svc.decision_function(x_val)

    y_pred_prob3, y_pred3 = linear_svc_model(x_train, y_train, x_val)

    ## Gradient Boosting Classifier
    def gradient_boost_model(x_train, y_train, x_val):
        gbc = GradientBoostingClassifier(n_estimators = 50, learning_rate = 1.0, max_depth=1, random_state = 0)
        gbc.fit(x_train, y_train)

        y_pred_prob4 = gbc.predict_proba(x_val)[:,1]
        y_pred4 = gbc.predict(x_val)
        return y_pred_prob4, y_pred4
    
    y_pred_prob4, y_pred4 = gradient_boost_model(x_train, y_train, x_val)

    #2. Aggregate Predictions
    predictions = np.stack((y_pred1, y_pred2, y_pred3, y_pred4), axis=1)
    fraud_votes = np.sum(predictions, axis=1)
    final_predictions = (fraud_votes >=3).astype(int)

    #3. Generate ROC-AUC Curve
    y_pred_prob = np.meain([y_pred_prob1, y_pred_prob2, y_pred_prob3, y_pred_prob4], axis=0)

    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    roc_auc = roc_auc_score(y_val, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # 5. Confusion Matrix and Classification Report
    conf_matrix = confusion_matrix(y_val, final_predictions)
    conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not Fraudulent", "Fraudulent"])

    plt.figure(figsize=(8, 6))
    conf_matrix_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    print(classification_report(y_val, final_predictions, target_names=['Not Fraud', 'Fraud']))



def main():
    folder = "Dataset_Health_Insurance"
    x_train, y_train, x_test, y_test, x_val, y_val = data_wrangle(folder)
    x_train, x_val, x_test, y_train = standardization_and_sample_balance(x_train, x_val, x_test, y_train)
    ensemble_model(x_train, y_train, x_test, y_test, x_val, y_val)

if __name__ == "__main__":
    main()