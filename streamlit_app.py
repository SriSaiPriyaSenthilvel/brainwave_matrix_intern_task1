import streamlit as st
import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Set Streamlit page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    try:
        true_df = pd.read_csv("True_reduced_final.csv")
        fake_df = pd.read_csv("Fake_reduced.csv")
        
        # Assign labels
        true_df["label"] = 1  # Real news
        fake_df["label"] = 0  # Fake news
        
        # Combine datasets
        data = pd.concat([true_df, fake_df], axis=0)
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data
        
        return data, true_df, fake_df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None

data, true_df, fake_df = load_data()

if data is not None:
    # Preprocessing function
    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text

    # Ensure no NaN values in text
    data["text"] = data["title"].fillna('') + " " + data["text"].fillna('')
    data["text"] = data["text"].apply(clean_text)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42
    )

    # Train model (if .pkl doesn't exist, train and save)
    @st.cache_resource
    def load_or_train_model():
        try:
            with open("fake_news_model.pkl", "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            model = make_pipeline(TfidfVectorizer(), MultinomialNB())
            model.fit(X_train, y_train)
            with open("fake_news_model.pkl", "wb") as f:
                pickle.dump(model, f)
            return model

    model = load_or_train_model()

    # Streamlit UI
    st.sidebar.image("fake.png", use_container_width=True)  # Add an image in the navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Dataset"])

    if page == "Home":
        st.title("📰 Fake News Detection")
        st.write("Unravel the truth! Enter a news article below to verify its authenticity.")

        user_input = st.text_area("Enter news text here:")
        if st.button("Check News"):
            if user_input.strip():
                prediction = model.predict([clean_text(user_input)])[0]
                result = "✅ Real News" if prediction == 1 else "❌ Fake News"
                st.subheader(result)
            else:
                st.warning("Please enter some text.")

    elif page == "Dataset":
        st.title("📊 Explore Dataset")
        st.write("Here are some real and fake news samples from the dataset:")
        
        st.write("### 🟢 Real News")
        st.dataframe(true_df.sample(min(5, len(true_df))))  # Prevents error if less than 5 samples
        
        st.write("### 🔴 Fake News")
        st.dataframe(fake_df.sample(min(5, len(fake_df))))  # Prevents error if less than 5 samples
