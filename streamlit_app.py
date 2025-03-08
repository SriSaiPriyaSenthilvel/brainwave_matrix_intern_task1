import streamlit as st
import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Set Streamlit page config (MUST BE FIRST)
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load dataset
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Label data
true_df["label"] = 1  # Real news
fake_df["label"] = 0  # Fake news

# Combine data
data = pd.concat([true_df, fake_df], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

data["text"] = data["title"] + " " + data["text"]  # Combine title and content
data["text"] = data["text"].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# Train model (if .pkl doesn't exist, train and save)
try:
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    with open("fake_news_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Load trained model
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI
st.sidebar.image("fake.png", use_container_width=True)  # Add an image in the navigation bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset"])

if page == "Home":
    st.title("📰 Fake News Detection")
    st.write("Unravel the truth! Enter a news article below to verify its authenticity.")
    
    user_input = st.text_area("Enter news text here:")
    if st.button("Check News"):
        if user_input:
            prediction = model.predict([clean_text(user_input)])[0]
            result = "✅ Real News" if prediction == 1 else "❌ Fake News"
            st.subheader(result)
        else:
            st.warning("Please enter some text.")

elif page == "Dataset":
    st.title("📊 Explore Dataset")
    st.write("Here are some real and fake news samples from the dataset:")
    st.write("### 🟢 Real News")
    st.dataframe(true_df.sample(5))
    st.write("### 🔴 Fake News")
    st.dataframe(fake_df.sample(5))
