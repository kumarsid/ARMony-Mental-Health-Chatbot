import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get the directory of the current script
script_dir = r"C:\Users\SidK\Documents\Documents\HSMA\Hackathon"

# Define file paths (loading from the parent directory)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

# Load the model, vectorizer, and label encoder
model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(encoder_path)

# Define a function to classify text
def classify_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# Define comforting messages and URLs
def get_message_and_url(label):
    messages = {
        'Anxiety': ("It's perfectly okay to feel anxious. Remember, you’re not alone and there are ways to manage these feelings. Take a moment for yourself, breathe deeply, and consider speaking with a mental health professional if needed.", "https://www.nhs.uk/conditions/anxiety/"),
        'Normal': ("It seems like you are doing well. Keep up the good work in maintaining your mental health. Remember to continue taking care of yourself and reach out if you need support.", "https://www.nhs.uk/conditions/mental-health/"),
        'Depression': ("Feeling down can be really tough, but you don’t have to go through it alone. Reach out to a trusted friend or mental health professional who can offer support and guidance. There is hope and help available.", "https://www.nhs.uk/conditions/depression/"),
        'Suicidal': ("If you are feeling overwhelmed or thinking about suicide, it’s crucial to seek immediate help. Contact a mental health professional, call emergency services, or reach out to a crisis hotline. Your safety and well-being are paramount.", "https://www.nhs.uk/conditions/suicide/"),
        'Stress': ("Feeling stressed is a common experience, but it’s important to manage it effectively. Try engaging in relaxation techniques, talking to a friend, or seeking professional advice to help alleviate stress. You don’t have to handle it all on your own.", "https://www.nhs.uk/every-mind-matters/mental-health-issues/stress/"),
        'Bipolar': ("Managing bipolar disorder can be challenging, but following your treatment plan and staying in touch with your healthcare provider is crucial. Remember, you’re not alone, and support is available.", "https://www.nhs.uk/conditions/bipolar-disorder/"),
        'Personality disorder': ("Dealing with personality disorders can be complex, but with the right support and therapy, it’s possible to manage your symptoms effectively. Reach out to mental health professionals who can provide tailored support.", "https://www.nhs.uk/conditions/personality-disorder/")
    }
    return messages.get(label, ("We're here to help. Please reach out to a mental health professional.", "https://www.nhs.uk"))

# Streamlit app layout
st.set_page_config(page_title="ARMony Healthbot 💖", page_icon="💖", layout="wide")

# Custom CSS
st.markdown(f"""
    <style>
    .big-font {{
        font-size: 30px !important;
        color: #4CAF50;
    }}
    .stButton>button {{
        background-color: #8B80F9;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #8B80F9;
    }}
    .message {{
        font-size: 20px;
        color: #3DDC97; /* Teal color */
        font-weight: bold;
    }}    
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("ARMony Healthbot 💖")
st.markdown("<div class='big-font'>Your mental health matters!</div>", unsafe_allow_html=True)

# Initialize session state for results and input
if 'result' not in st.session_state:
    st.session_state.result = None
if 'message' not in st.session_state:
    st.session_state.message = None
if 'url' not in st.session_state:
    st.session_state.url = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Initialize session state for results and input
if 'result' not in st.session_state:
    st.session_state.result = None
if 'message' not in st.session_state:
    st.session_state.message = None
if 'url' not in st.session_state:
    st.session_state.url = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Use a form for user input and classification
with st.form(key='classification_form'):
    user_input = st.text_area("Enter a statement:", key="input_text", value=st.session_state.user_input)
    submit_button = st.form_submit_button("Classify")

    if submit_button:
        if user_input:
            st.session_state.result = classify_text(user_input)
            st.session_state.message, st.session_state.url = get_message_and_url(st.session_state.result)
            st.session_state.user_input = user_input  # Store the input in session state
        else:
            st.warning("Please enter a statement to classify.")

# Display results if available
if st.session_state.result:
    st.success(st.session_state.message)
    st.markdown(f"[More Information]({st.session_state.url})")

    # Button to reset input field and state
    if st.button("Submit New Response"):
        # Clear session state
        st.session_state.result = None
        st.session_state.message = None
        st.session_state.url = None
        st.session_state.user_input = ""
        # Reset the text area widget value by updating session state
        # This will effectively clear the form when the button is clicked
        st.rerun()  # Use this line if you need to reset the entire app state, otherwise use only the session state updates
