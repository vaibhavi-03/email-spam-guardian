# 🛡️ Email Spam Guardian

An end-to-end machine learning project that classifies emails as **spam** or **legitimate**,
served through a live, interactive Streamlit dashboard.

**Live app:** _add your Streamlit Community Cloud URL here after deployment_

---

## Problem statement

Given structured signals about an email — length, number of links, special characters,
capitalised words, whether it has an attachment, the sender's domain, and its subject
category — predict whether it is spam.

## Dataset

`data/email_spam_detection.csv` — 7,000 emails, 9 columns. The raw data has realistic
messiness that mirrors real-world logging issues: a blank sender field, missing
`Email_Length` values, and two sentinel values standing in for "not recorded"
(`Num_Special_Chars == -1`, `Num_Links == 99`). All of this is cleaned inside the notebook
before any modelling happens.

## Project structure

```
email-spam-guardian/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── notebooks/
│   └── email-spam-detection.ipynb
├── data/
│   └── email_spam_detection.csv
├── models/
│   ├── spam_pipeline.joblib
│   ├── metrics.json
│   └── feature_importance.json
└── assets/
    └── style.css                
```
## How it works

1. **`notebooks/email-spam-detection.ipynb`** — loads the raw CSV, cleans it (fixes the
   blank sender field, treats sentinel values as missing, imputes per-class medians),
   explores it visually, engineers a few extra features (`email_domain`,
   `subject_length`, `is_urgent_subject`), trains and compares four models (Logistic
   Regression, Decision Tree, Random Forest, Naive Bayes) with 5-fold cross-validation,
   and saves the best pipeline to `models/`.
2. **`app.py`** — loads `models/spam_pipeline.joblib`, applies the same feature
   engineering used during training, and serves a live scan page where you can enter an
   email's details and get an instant spam verdict with its probability.

## Model performance

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|---------:|----------:|-------:|------:|--------:|
| Logistic Regression  | ~0.999  | ~0.995    | 1.000  | ~0.998| 1.000   |
| Random Forest        | ~0.999  | ~0.998    | 1.000  | ~0.999| 1.000   |
| Decision Tree        | ~0.996  | ~0.993    | ~0.995 | ~0.994| ~0.996  |
| Naive Bayes          | ~0.999  | ~0.995    | 1.000  | ~0.998| 1.000   |

**Honest caveat:** this dataset is synthetic, and its numeric features were generated to
separate the two classes very cleanly — that's why every model scores near-perfectly. Real
spam is adversarial and doesn't cooperate like this. Worth saying plainly if this comes up
in an interview: understanding *why* a benchmark looks too good matters more than the
number itself.


```

## Deployment

1. Push this repo to GitHub (public, so Streamlit Community Cloud's free tier can reach it).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click
   **Create app**.
3. Point it at your repo, branch `main`, file `app.py`. Click **Deploy**.
4. Once live, replace the placeholder URL at the top of this README, commit, and push.

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib · seaborn

## Author

Vaibhavi — [GitHub: vaibhavi-03](https://github.com/vaibhavi-03)
