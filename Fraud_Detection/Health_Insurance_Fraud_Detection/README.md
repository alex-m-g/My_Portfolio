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
This project aims to " predict the potentially fraudulent providers " based on the claims filed by them. We will also discover important variables that help detect the behavior of potential fraud providers. Further, we will study fraudulent patterns in the provider's claims to understand the future behaviour of providers.

## Analysis Error (8/21/2024)
The Jupyter file Health_Insurance_Fraud_Detection.ipyb was the first try at data wrangling and analysis. However, a ROC-AUC curve analysis showed poor True Positive Rates. Will need to re-analyze and re-wrangle the data.

## How To Improve AUC?
To improve AUC, it is important to improve the classifier's performance. Several measures could be taken for experimentation. However, it will depend on the problem and the data to decide which measure will work. 

(1) Feature normalization and scaling. Basically, this method improves the performance of the linear (logistic) model. 

(2) Improve class imbalance. In classification problems, a bunch of them have imbalanced classes. Setting class weights or performing upward/downward sampling will help. 

(3) Optimize other scores. Defining the right score for the problem and optimizing the score will help the prediction performance. 

(4) Explore different models. Choose the model with the best performance on the problem among the classification models. 

(5) Tune the parameter through grid search. Grid search is an automatic way to tune your parameter.

(6) Error analysis. Review the false positive and false negative cases and find the reasons for this. 

(7) Include more features or fewer features. 

(8) There is also research on optimizing AUC scores directly by investigating the relationship between AUC and error rate or with the models, leading to a more straightforward but also more complicated analysis. 
