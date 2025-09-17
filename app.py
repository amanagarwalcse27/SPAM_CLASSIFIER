import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from  preprocess import full_pipeline   # apna preprocessing function import kar liya
import nltk 
nltk.download('punkt')

#Load trained Model and Vectorizer
model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))


# Streamlit UI
st.title("📩 SPAM CLASSIFIER APP")

#Input Container
user_input = st.text_area("Enter the message")

# Button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Step 1: Preprocess the input
        transformed_text = full_pipeline(user_input)

        # Step 2: Convert into vector form
        vector_input = tfidf.transform([transformed_text]).toarray()

        # Step 3: Predict using trained model
        prediction = model.predict(vector_input)[0]

        # Step 4: Show result
        if prediction == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")




