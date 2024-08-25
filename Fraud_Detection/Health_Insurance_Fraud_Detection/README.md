# Health Insurance Fraud Detection

Provider Fraud is one of the biggest problems facing Medicare. According to the government, the total Medicare spending increased exponentially due to fraud in Medicare claims. Healthcare fraud is an organized crime that involves peers of providers, physicians, and beneficiaries acting together to make fraud claims.

Rigorous analysis of Medicare data has yielded many physicians who indulge in fraud. They adopt ways in which an ambiguous diagnosis code is used to adopt the costliest procedures and drugs. Insurance companies are the most vulnerable institutions impacted due to these bad practices. Due to this reason, insurance companies increased their insurance premiums and as result healthcare is becoming costly matter day by day.

Healthcare fraud and abuse take many forms. Some of the most common types of fraud by providers are:

a) Billing for services that were not provided.

b) Duplicate submission of a claim for the same service.

c) Misrepresenting the service provided.

d) Charging for a more complex or expensive service than was actually provided.

e) Billing for a covered service when the service actually provided was not covered.

Problem Statement
This project aims to " predict the potentially fraudulent providers " based on the claims filed by them. We will also discover important variables that help detect the behavior of potential fraud providers. Further, we will study fraudulent patterns in the provider's claims to understand the future behavior of providers.

## Health_Insurance_Fraud_Detection_1
This was my first exposure to the dataset. The data-wrangling mistake was removing data points instead of drawing inferences from data such as DOB, time admitted, time discharged, etc. The trained models poorly detected fraudulent transactions.

## Health_Insurance_Fraud_Detection_2
Second run into data wrangling, I added more data (columns) to the dataset to provide numerical, adequate data for the models to be better trained. After utilizing some data to create more data numerically. I removed the 'no-longer-needed' data. The trained models significantly improved. However, I believe the models can improve if I balance the data.

## Health_Insurance_Fraud_Detection_3
The data wrangling technique stayed the same but I then removed some of the oversampled data 'Non Fraudulent' data points to match the count with 'Fraudulent' data points. The trained models significantly improved in catching fraudulent transactions; however, the ROC-AUC score decreased due to the low data volume resulting from undersampling. I believe the model can be improved by utilizing alternative methods to combat imbalanced data without undersampling and eliminating too many data points.

## Health_Insurance_Fraud_Detection_4
The data wrangling technique stayed the same, but I used the SMOTE oversampling technique to create emulated undersampled data points to equalize the dataset. The trained models did better at detecting fraudulent transactions; However, it called too many Non-Fraudulent transactions Fraudulent. This requires further fitting of the SMOTE resampling technique to optimize the emulated data to train the models appropriately.
