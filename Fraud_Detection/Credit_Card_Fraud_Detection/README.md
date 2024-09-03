### Credit Card Fraud Detection Pipeline

**Project Overview:**
Developed a credit card fraud detection pipeline using an ensemble of machine learning models, including Linear Support Vector Classifier (LinearSVC), Logistic Regression, Gradient Boosting Classifier, and a shallow neural network. The pipeline is designed to identify fraudulent transactions with high accuracy by aggregating predictions from multiple models.

**Key Contributions:**
- **Data Preprocessing:** Utilized `Pandas` and `NumPy` to handle data wrangling, including scaling features with `RobustScaler`, normalizing the `Amount` and `Time` columns, and addressing class imbalance by balancing fraudulent and non-fraudulent transactions.
- **Model Training:** Trained multiple models (`LinearSVC`, `Logistic Regression`, `Gradient Boosting Classifier`, and a shallow neural network built with `Keras` and `TensorFlow`), fine-tuning them to improve detection accuracy.
- **Ensemble Modeling:** Implemented an ensemble approach to aggregate predictions, using majority voting to label transactions as fraudulent if at least three models predicted fraud.
- **Validation and Evaluation:** Split the dataset into training, testing, and validation sets, ensuring a robust evaluation of model performance. Generated detailed classification reports to evaluate precision, recall, and F1-score for each model.
- **Fraud Detection:** Successfully identified and reported fraudulent transactions, highlighting the effectiveness of the ensemble approach in minimizing false positives and maximizing detection accuracy.

**Technologies Used:**
- **Programming Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow, Keras
- **Tools:** ModelCheckpoint (for saving the best neural network model), `classification_report` (for model evaluation)

This description provides a comprehensive overview of the project, highlights your contributions, and lists the technologies you used.
## Dataset
CSV File: [Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv)
