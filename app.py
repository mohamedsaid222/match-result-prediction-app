import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =======================
# 1. Load & Train Model
# =======================
@st.cache_resource
def load_and_train():
    df = pd.read_csv("results.csv")

    # Create match result column
    def match_result(row):
        if row['home_score'] > row['away_score']:
            return "Home Win"
        elif row['home_score'] < row['away_score']:
            return "Away Win"
        else:
            return "Draw"

    df['result'] = df.apply(match_result, axis=1)

    # Encode teams
    home_encoder = LabelEncoder()
    away_encoder = LabelEncoder()
    df['home_team_enc'] = home_encoder.fit_transform(df['home_team'])
    df['away_team_enc'] = away_encoder.fit_transform(df['away_team'])

    X = df[['home_team_enc', 'away_team_enc']]
    y = df['result']

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, home_encoder, away_encoder, df

model, home_encoder, away_encoder, df = load_and_train()

# =======================
# 2. Streamlit UI
# =======================
st.title("⚽ Match Result Prediction")
st.write("Enter two teams to predict the match outcome (Home / Away / Draw).")

# Dropdowns for teams
teams = sorted(df['home_team'].unique())
home_team = st.selectbox("🏟️ Home Team", teams)
away_team = st.selectbox("✈️ Away Team", teams)

if st.button("Predict"):
    home_encoded = home_encoder.transform([home_team])[0]
    away_encoded = away_encoder.transform([away_team])[0]

    pred = model.predict([[home_encoded, away_encoded]])[0]
    prob = model.predict_proba([[home_encoded, away_encoded]])[0]

    st.success(f"Predicted Result: **{pred}**")
    st.write("### Probabilities:")
    for cls, p in zip(model.classes_, prob):
        st.write(f"- {cls}: {p:.2f}")
