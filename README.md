# Spam SMS Detection

This is a basic machine learning project for detecting spam SMS messages.

The model takes an SMS message as input and predicts whether it is **Spam** or **Ham**.

## About the Project

In this project, I used a spam SMS dataset and trained a simple text classification model.
The text data is cleaned first, then converted into numerical form using TF-IDF. After that, I trained a Naive Bayes model to classify the messages.

I also made a small Streamlit app so that we can enter any message and check the prediction.

## What I Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Joblib

## Steps Done

* Loaded the dataset
* Kept only the message and label columns
* Cleaned the text messages
* Converted Spam/Ham labels into numbers
* Split the data into training and testing data
* Used TF-IDF vectorizer
* Trained a Multinomial Naive Bayes model
* Checked model accuracy and classification report
* Saved the model and vectorizer
* Created a simple Streamlit app

## Model

I used **Multinomial Naive Bayes** for this project.
It is a simple and commonly used model for text classification problems like spam detection.
