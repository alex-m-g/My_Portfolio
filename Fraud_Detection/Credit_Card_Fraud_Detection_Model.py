import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

def ensemble_model(x_train_b, y_train_b, x_test_b, y_test_b, original_df):
    # 1. Train the models
    ## Linear SVC Model
    svc_b = LinearSVC(class_weight='balanced')
    svc_b.fit(x_train_b, y_train_b)
    classification_report(y_test_b, svc_b.predict(x_test_b), target_names=['Not Fraud', 'Fraud'])

    ## Logistic Regression Model
    logistic_model_b = LogisticRegression()
    logistic_model_b.fit(x_train_b, y_train_b)
    logistic_model_b.score(x_train_b, y_train_b)
    classification_report(y_test_b, logistic_model_b.predict(x_test_b), target_names=['Not Fraud', 'Fraud'])

    ## Shallow Neural Network Model
    shallow_nn_b = Sequential()
    shallow_nn_b.add(InputLayer((x_train_b.shape[1],)))
    shallow_nn_b.add(Dense(2, 'relu'))
    shallow_nn_b.add(BatchNormalization())
    shallow_nn_b.add(Dense(1, 'sigmoid'))
    checkpoint = ModelCheckpoint('shallow_nn_b.keras', save_best_only=True)
    shallow_nn_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_test_b, y_test_b), epochs=40, callbacks=checkpoint)
    classification_report(y_test_b, (shallow_nn_b.predict(x_test_b).flatten() > 0.5).astype(int), target_names=['Not Fraud', 'Fraud'])

    ## Gradient Boosting Classifier
    gbc_b = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    gbc_b.fit(x_train_b, y_train_b)
    classification_report(y_test_b, gbc_b.predict(x_test_b), target_names=['Not Fraud', 'Fraud'])


    # 2. Make predictions on the validation set using each model
    y_pred_svc = svc_b.predict(x_test_b)
    y_pred_logistic = logistic_model_b.predict(x_test_b)
    y_pred_nn = (shallow_nn_b.predict(x_test_b) > 0.5).astype(int).flatten()  # Thresholding at 0.5 for binary classification
    y_pred_gbc = gbc_b.predict(x_test_b)


    # 3. Aggregate the predictions
    # Stack predictions to form a (number_of_samples, 4) array
    predictions = np.stack((y_pred_svc, y_pred_logistic, y_pred_nn, y_pred_gbc), axis=1)

    # Sum across the models to see how many labeled each instance as "Fraud"
    fraud_votes = np.sum(predictions, axis=1)

    # Label as "Fraud" (1) if 3 or more models voted "Fraud", else "Not Fraud" (0)
    final_predictions = (fraud_votes >= 3).astype(int)

    # 4. Generate the classification report
    print(classification_report(y_test_b, final_predictions, target_names=['Not Fraud', 'Fraud']))

    # 5. Identify Fraudulent Transactions
    anomalies = np.where(final_predictions ==1)[0] # locates the transactions labeled "1" (Fradulent)
    anomaly_indices = original_df.iloc[anomalies + len(y_train_b)]
    return anomaly_indices


## Import Data ##
df = pd.read_csv('creditcard.csv')

## Data Wrangling ##
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

anomalies = ensemble_model(x_train_b, y_train_b, x_test_b, y_test_b, balanced_df2)

print("Fraudulent Transactions Detected:")
print(anomalies)