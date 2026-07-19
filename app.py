import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Email Spam Guardian", page_icon="🛡️", layout="wide")

CSS_PATH = Path("assets/style.css")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Feature engineering — same logic used to train the model, kept in sync here.
# (If you split this into features.py per the earlier guide, swap this block
# for: from features import add_engineered_columns, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
# ---------------------------------------------------------------------------
URGENT_SUBJECTS = {"Security Alert", "Account Verification", "Win Prize"}
CATEGORICAL_FEATURES = ["email_domain", "Subject"]
NUMERIC_FEATURES = [
    "Email_Length", "Num_Links", "Num_Special_Chars", "Capital_Words",
    "Has_Attachment", "subject_length", "is_urgent_subject",
]
KNOWN_SUBJECTS = [
    "Meeting", "Security Alert", "Win Prize", "Invoice",
    "Account Verification", "Project Update", "Offer", "Greetings",
]

# Keep UI inputs inside the range represented by the training data.  The
# StandardScaler is already part of ``spam_pipeline.joblib`` and runs during
# ``predict_proba``; these bounds prevent the UI from feeding it unrealistic
# values (the training range for Email_Length was 20–265 characters).
EMAIL_LENGTH_MIN = 20
EMAIL_LENGTH_MAX = 265
EMAIL_LENGTH_DEFAULT = 95  # training-data median
EMAIL_LENGTH_HIGH = 158    # training-data 75th percentile


def add_engineered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["email_domain"] = frame["Sender_Email"].apply(
        lambda x: x.split("@")[-1] if isinstance(x, str) and "@" in x else "unknown"
    )
    frame["subject_length"] = frame["Subject"].astype(str).str.len()
    frame["is_urgent_subject"] = frame["Subject"].isin(URGENT_SUBJECTS).astype(int)
    return frame


@st.cache_resource
def load_pipeline():
    return joblib.load(MODELS_DIR / "spam_pipeline.joblib")


@st.cache_data
def load_feature_importance():
    path = MODELS_DIR / "feature_importance.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-row">
            <div class="hero-icon">🛡️</div>
            <div>
                <h1>Email Spam Guardian</h1>
                <p>Enter an email's details below — the model flags whether it looks like spam,
                and shows exactly which signals drove that call.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None

col_left, col_right = st.columns([1.15, 1], gap="large")

# ---------------------------------------------------------------------------
# Left column — input form
# ---------------------------------------------------------------------------
with col_left:
    with st.container(border=True):
        st.markdown("#### 📨 Email details")

        sender_email = st.text_input("Sender email", placeholder="promo@offers-daily.net")
        subject = st.selectbox("Subject", KNOWN_SUBJECTS, index=2)

        c1, c2 = st.columns(2)
        with c1:
            email_length = st.slider(
                "Email length (characters)",
                EMAIL_LENGTH_MIN,
                EMAIL_LENGTH_MAX,
                EMAIL_LENGTH_DEFAULT,
                help="Use the email body length. Values are limited to the range used to train the model.",
            )
            num_special_chars = st.slider("Special characters", 0, 50, 3)
        with c2:
            num_links = st.slider("Number of links", 0, 20, 1)
            capital_words = st.slider("ALL-CAPS words", 0, 30, 2)

        has_attachment = st.checkbox("Has an attachment")

        scan_clicked = st.button("🔍 Check this email", use_container_width=True)

        if scan_clicked:
            row = pd.DataFrame([{
                "Sender_Email": sender_email or "unknown@unknown.com",
                "Subject": subject,
                "Email_Length": float(email_length),
                "Num_Links": float(num_links),
                "Num_Special_Chars": float(num_special_chars),
                "Capital_Words": capital_words,
                "Has_Attachment": int(has_attachment),
            }])
            row_feat = add_engineered_columns(row)[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
            proba = float(pipeline.predict_proba(row_feat)[0, 1])

            st.session_state.result = {
                "proba": proba,
                "email_length": email_length,
                "num_links": num_links,
                "num_special_chars": num_special_chars,
                "capital_words": capital_words,
                "has_attachment": has_attachment,
            }

# ---------------------------------------------------------------------------
# Right column — verdict
# ---------------------------------------------------------------------------
with col_right:
    result = st.session_state.result

    if result is None:
        st.markdown(
            """
            <div class="placeholder-card">
                <div class="big-icon">📭</div>
                <p>Fill in the email details and click <b>Check this email</b><br>
                to see the verdict here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        proba = result["proba"]
        is_spam = proba >= 0.5
        verdict_class = "verdict-spam" if is_spam else "verdict-safe"
        icon = "🚨" if is_spam else "✅"
        title = "Likely Spam" if is_spam else "Looks Legitimate"

        st.markdown(
            f"""
            <div class="verdict-card {verdict_class}">
                <div class="verdict-icon">{icon}</div>
                <div class="verdict-title">{title}</div>
                <div class="verdict-sub">Spam probability: {proba*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"color": "#eaf2f4"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9fb3bf"},
                "bar": {"color": "#e94560" if is_spam else "#17c3b2"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(23,195,178,0.15)"},
                    {"range": [50, 100], "color": "rgba(233,69,96,0.15)"},
                ],
            },
        ))
        gauge.update_layout(
            height=200, margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#eaf2f4"},
        )
        st.plotly_chart(gauge, use_container_width=True)

        # Signal chips — quick plain-language read of what stood out
        chips = []
        chips.append(("Links: high" if result["num_links"] >= 4 else "Links: normal",
                       "risk" if result["num_links"] >= 4 else "ok"))
        chips.append(("Special chars: high" if result["num_special_chars"] >= 8 else "Special chars: normal",
                       "risk" if result["num_special_chars"] >= 8 else "ok"))
        chips.append(("ALL-CAPS: heavy" if result["capital_words"] >= 6 else "ALL-CAPS: light",
                       "risk" if result["capital_words"] >= 6 else "ok"))
        chips.append(("Long email" if result["email_length"] >= EMAIL_LENGTH_HIGH else "Typical length",
                       "risk" if result["email_length"] >= EMAIL_LENGTH_HIGH else "ok"))

        chip_html = "".join(f'<span class="chip {cls}">{label}</span>' for label, cls in chips)
        st.markdown(f'<div class="chip-row">{chip_html}</div>', unsafe_allow_html=True)

        
st.markdown('<div class="footer-note">Email Spam Guardian · Built with Streamlit</div>', unsafe_allow_html=True)
