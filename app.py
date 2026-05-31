import streamlit as st
import joblib
import re


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.set_page_config(
    page_title="Spam SMS Detection",
    layout="centered"
)

st.title("Spam SMS Detection")

st.write(
    "This app classifies SMS messages as Spam or Ham using "
    "TF-IDF vectorization and a Multinomial Naive Bayes classifier."
)

message = st.text_area("Enter an SMS message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned_message = clean_text(message)
        vectorized_message = vectorizer.transform([cleaned_message])
        prediction = model.predict(vectorized_message)[0]

        if prediction == 1:
            st.error("This message is predicted as SPAM.")
        else:
            st.success("This message is predicted as HAM / Not Spam.")