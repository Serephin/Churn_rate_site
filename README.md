# Churn_rate_site
A simple FastAPI for Churn Rate prediction 
Paste https://churn-rate-site.onrender.com/docs to search bar. The fastapi swagger will be openned) 

Project Overview

This project provides a FastAPI service for predicting the probability of a client leaving the bank (churn).
The API exposes endpoints for sending client data and receiving a predicted probability of churn.
The model is trained on a dataset of 15,000 clients with 14 features.

Features

Predict probability of client churn.

Preprocessing pipeline included for categorical and numerical features.

Model selection based on XGBoost, optimized for recall to capture as many churned clients as possible.

Stratified train/test split and class-weight handling for imbalanced classes.
