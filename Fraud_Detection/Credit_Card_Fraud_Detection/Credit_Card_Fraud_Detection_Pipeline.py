import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def ensemble_model(x_train_b, y_train_b, x_test_b, y_test_b, x_val_b, y_val_b):
    # 1. Train the models
    ## Linear SVC Model
    svc_b = LinearSVC(class_weight='balanced')
    svc_b.fit(x_train_b, y_train_b)

    ## Logistic Regression Model
    logistic_model_b = LogisticRegression()
    logistic_model_b.fit(x_train_b, y_train_b)

    ## Shallow Neural Network Model
    shallow_nn_b = Sequential()
    shallow_nn_b.add(InputLayer((x_train_b.shape[1],)))
    shallow_nn_b.add(Dense(2, 'relu'))
    shallow_nn_b.add(BatchNormalization())
    shallow_nn_b.add(Dense(1, 'sigmoid'))
    checkpoint = ModelCheckpoint('shallow_nn_b.keras', save_best_only=True)
    shallow_nn_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_test_b, y_test_b), epochs=40, callbacks=checkpoint)

    ## Gradient Boosting Classifier
    gbc_b = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    gbc_b.fit(x_train_b, y_train_b)

    # 2. Make predictions on the validation set using each model
    y_pred_svc = svc_b.predict(x_val_b)
    y_pred_logistic = logistic_model_b.predict(x_val_b)
    y_pred_nn = (shallow_nn_b.predict(x_val_b) > 0.5).astype(int).flatten()
    y_pred_gbc = gbc_b.predict(x_val_b)

    # 3. Aggregate the predictions
    predictions = np.stack((y_pred_svc, y_pred_logistic, y_pred_nn, y_pred_gbc), axis=1)
    fraud_votes = np.sum(predictions, axis=1)
    final_predictions = (fraud_votes >= 3).astype(int)

    # 4. Generate ROC-AUC Curve
    y_val_pred_prob = np.mean([svc_b.decision_function(x_val_b),
                               logistic_model_b.predict_proba(x_val_b)[:, 1],
                               shallow_nn_b.predict(x_val_b).flatten(),
                               gbc_b.predict_proba(x_val_b)[:, 1]], axis=0)
    
    fpr, tpr, thresholds = roc_curve(y_val_b, y_val_pred_prob)
    roc_auc = roc_auc_score(y_val_b, y_val_pred_prob)

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
    conf_matrix = confusion_matrix(y_val_b, final_predictions)
    conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not Fraudulent", "Fraudulent"])

    plt.figure(figsize=(8, 6))
    conf_matrix_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    print(classification_report(y_val_b, final_predictions, target_names=['Not Fraud', 'Fraud']))

def data_wrangle(df):
    df2 = df.copy() # Copy the original DataFrame
    df2['Amount'] = RobustScaler().fit_transform(df2['Amount'].to_numpy().reshape(-1,1)) # Normalize the 'Amount' column to reduce influence of outliers.
    time = df2['Time']
    df2['Time'] = (time - time.min()) / (time.max() - time.min()) # Normalize the 'Time' column by scaling it to range [0,1]
    df2 = df2.sample(frac=1) # Shuffle the DataFrame rows to ensure randomness in the dataset
    not_frauds = df2.query('Class == 0')
    frauds = df2.query('Class == 1')
    print("Instances before Balance: ",df2['Class'].value_counts()) # check the number of instances of 'Frauds' and 'Non-Frauds' to check class imbalance.
    balanced_df2 = pd.concat([frauds, not_frauds.sample(len(frauds))]) # Creates a balanced dataset by same number of Non-Fradulent transactions as there are fraudulent ones.
    print("Instances after balance: ",balanced_df2['Class'].value_counts()) # verify the class distribution to ensure data balance.
    # There is 492 Fraud cases and 492 non-Fraud cases. Total of 984 rows.
    balanced_df2 = balanced_df2.sample(frac=1) # shuffle the balanced dataset to ensure randomness after concatenation.
    balanced_df2_np = balanced_df2.to_numpy() # Convert the balanced DataFrame into NumPy for model training.

    # Split the dataset into training, testing, and validation sets
    # Total of 984 Transations - Training Set: 70% (689), Test Set: 15% (148), Validation Set: 15% (147)
    x_train_b, y_train_b = balanced_df2_np[:689, :-1], balanced_df2_np[:689, -1].astype(int) # The first 700 samples used for training
    x_test_b, y_test_b = balanced_df2_np[689:837, :-1], balanced_df2_np[689:837, -1].astype(int) # The next 142 samples are used for testing
    x_val_b, y_val_b = balanced_df2_np[837:, :-1], balanced_df2_np[837:, -1].astype(int) # The remaining samples are used for validation

    # Print the shapes of the training, testing, and validation sets to verify correct splitting
    print("Shapes: ",x_train_b.shape, y_train_b.shape, x_test_b.shape, y_test_b.shape, x_val_b.shape, y_val_b.shape)

    # Count and display the number of instances in each class for training, testing, and validation sets
    print("Count and Number of Instances: ", pd.Series(y_train_b).value_counts(), pd.Series(y_test_b).value_counts(), pd.Series(y_val_b).value_counts())

    return x_train_b, y_train_b, x_test_b, y_test_b, x_val_b, y_val_b

def main():
    df = pd.read_csv('creditcard.csv')
    x_train_b, y_train_b, x_test_b, y_test_b, x_val_b, y_val_b = data_wrangle(df)
    anomalies = ensemble_model(x_train_b, y_train_b, x_test_b, y_test_b, x_val_b, y_val_b)
    print(anomalies)

if __name__ == "__main__":
    main()