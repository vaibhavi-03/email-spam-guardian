import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Email Spam Guardian", page_icon="🛡️", layout="centered")

# top of app.py
URGENT_SUBJECTS = {"Security Alert", "Account Verification", "Win Prize"}

def add_engineered_columns(frame):
    frame = frame.copy()
    frame["email_domain"] = frame["Sender_Email"].apply(
        lambda x: x.split("@")[-1] if isinstance(x, str) and "@" in x else "unknown"
    )
    frame["subject_length"] = frame["Subject"].astype(str).str.len()
    frame["is_urgent_subject"] = frame["Subject"].isin(URGENT_SUBJECTS).astype(int)
    return frame

CATEGORICAL_FEATURES = ["email_domain", "Subject"]
NUMERIC_FEATURES = [
    "Email_Length", "Num_Links", "Num_Special_Chars", "Capital_Words",
    "Has_Attachment", "subject_length", "is_urgent_subject",
]

# Cache this so the model only loads once per session, not on every rerun
@st.cache_resource
def load_pipeline():
    return joblib.load(Path("models/spam_pipeline.joblib"))

pipeline = load_pipeline()

st.title("🛡️ Email Spam Guardian")
st.write("Enter an email's details and check whether it looks like spam.")

# --- Inputs ---
sender_email = st.text_input("Sender email", "promo@offers-daily.net")
subject = st.selectbox("Subject", [
    "Meeting", "Security Alert", "Win Prize", "Invoice",
    "Account Verification", "Project Update", "Offer", "Greetings",
])
email_length = st.slider("Email length (characters)", 0, 1000, 200)
num_links = st.slider("Number of links", 0, 20, 1)
num_special_chars = st.slider("Number of special characters", 0, 50, 3)
capital_words = st.slider("ALL-CAPS words", 0, 30, 2)
has_attachment = st.checkbox("Has an attachment")

# --- Predict ---
if st.button("Check this email"):
    row = pd.DataFrame([{
        "Sender_Email": sender_email, "Subject": subject,
        "Email_Length": email_length, "Num_Links": num_links,
        "Num_Special_Chars": num_special_chars, "Capital_Words": capital_words,
        "Has_Attachment": int(has_attachment),
    }])

    # same feature engineering function from your notebook — reuse it, don't rewrite it
    row_feat = add_engineered_columns(row)[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    proba = pipeline.predict_proba(row_feat)[0, 1]
    if proba >= 0.5:
        st.error(f"🚨 Likely spam — {proba*100:.1f}% probability")
    else:
        st.success(f"✅ Looks legitimate — {proba*100:.1f}% spam probability")
    st.progress(proba)