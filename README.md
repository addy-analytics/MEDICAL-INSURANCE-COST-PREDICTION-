## Medical Insurance Cost Prediction Using Machine Learning

This repository focuses on the application of machine learning to predict the costs of medical insurance, utilizing a dataset sourced from [Kaggle](https://www.kaggle.com/mirichoi0218/insurance?select=insurance.csv). The dataset incorporates features derived from individual and local health data, enabling the creation of predictive models to estimate insurance amounts across different categories of individuals. The dataset serves as a valuable resource for training machine learning models aimed at forecasting medical insurance costs based on diverse factors.

## Introduction
Medical insurance costs are critical considerations in healthcare planning and financial decision-making. Understanding the factors influencing these costs is essential for both insurers and policyholders. Machine learning provides a powerful toolset to analyze and predict medical insurance expenses based on diverse variables. In this context, predictive modeling techniques such as K-Nearest Neighbors (KNN), XGBoost, Linear Regression, and Random Forest Regression have emerged as effective means to estimate and forecast medical insurance costs. By leveraging patterns and relationships within datasets derived from individual and local health information, these models contribute to more accurate predictions, assisting in risk assessment, resource allocation, and overall decision-making within the healthcare insurance domain. This repository explores the application of these machine learning models to enhance our understanding and prediction of medical insurance costs, utilizing a dataset sourced from Kaggle that encapsulates various features relevant to insurance forecasting

## Methodology
Python Libraries
The libraries used on this project include:
- Pandas – For storing and manipulating structured data. Pandas functionality is built on NumPy
- Numpy – For multi-dimensional array, matrix data structures and, performing mathematical operations
- Scikit learn – For Machine learning tasks
- Seaborn and Matplotlib - for visualizing data
The main steps for this project can be summarized in the flow chart below:
![workflow](https://github.com/addy-analytics/Medical-Insurance-Cost-Prediction-Using-Machine-Learning/assets/107724453/378bc355-ad36-4eca-b58b-7d87b2294ab6)

## Final Results
XGBoost stands out as the most effective regression model, surpassing others which is deemed less optimal. Model gave 86% accuracy for Medical Insurance Amount Prediction using `XGBoost Regressor`. Individuals have the opportunity to compute their insurance expenses employing the  XGBoost model. Future enhancements could involve the development of a web application utilizing the XGBoost or Gradient Boosting algorithm, along with the utilization of a more extensive dataset compared to the one employed in the current study. This could further refine and advance the predictive capabilities of the model.
