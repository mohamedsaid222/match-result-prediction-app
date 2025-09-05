import streamlit as st
import pandas as pd
import numpy as np

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("results.csv", parse_dates=['date'])
    return df

df = load_data()

st.title("⚽ Match Result Prediction App")

# ======================
# Team Selection
# ======================
teams = pd.concat([df['home_team'], df['away_team']]).unique()
team = st.selectbox("اختر الفريق", sorted(teams))

# ======================
# Last 5 Matches Functions
# ======================
def last_five_matches(team, df):
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    return team_matches.sort_values('date', ascending=False).head(5)

def result_for(team, row):
    if row['home_score'] > row['away_score']:
        return 'Win' if row['home_team'] == team else 'Loss'
    elif row['home_score'] < row['away_score']:
        return 'Win' if row['away_team'] == team else 'Loss'
    else:
        return 'Draw'

# ======================
# Display Stats
# ======================
if team:
    st.subheader(f"📊 Last 5 Matches for {team}")

    recent = last_five_matches(team, df).copy()
    recent['Result'] = recent.apply(lambda x: result_for(team, x), axis=1)

    # جدول آخر 5 مباريات
    st.write("### Last 5 Results")
    st.table(recent[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'Result']])

    # ملخص (عدد فوز/تعادل/خسارة)
    summary = recent['Result'].value_counts()
    st.write("### Summary of Last 5 Matches")
    st.write(summary.to_frame().T)

    # رسم بياني
    st.write("### Visualization")
    st.bar_chart(summary)

# ======================
# TODO: Add your Prediction Model Code here
# ======================
st.write("🔮 Prediction model will appear here...")
