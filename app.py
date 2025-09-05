import streamlit as st
import pandas as pd

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
home_team = st.selectbox("اختر الفريق المستضيف (Home)", sorted(teams))
away_team = st.selectbox("اختر الفريق الضيف (Away)", sorted(teams))

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
# Display Stats for Both Teams
# ======================
if home_team and away_team and home_team != away_team:
    col1, col2 = st.columns(2)

    # Home team stats
    with col1:
        st.subheader(f"🏠 {home_team} - Last 5 Matches")
        recent_home = last_five_matches(home_team, df).copy()
        recent_home['Result'] = recent_home.apply(lambda x: result_for(home_team, x), axis=1)

        st.table(recent_home[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'Result']])
        summary_home = recent_home['Result'].value_counts()
        st.write("Summary")
        st.write(summary_home.to_frame().T)
        st.bar_chart(summary_home)

    # Away team stats
    with col2:
        st.subheader(f"✈️ {away_team} - Last 5 Matches")
        recent_away = last_five_matches(away_team, df).copy()
        recent_away['Result'] = recent_away.apply(lambda x: result_for(away_team, x), axis=1)

        st.table(recent_away[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'Result']])
        summary_away = recent_away['Result'].value_counts()
        st.write("Summary")
        st.write(summary_away.to_frame().T)
        st.bar_chart(summary_away)

# ======================
# TODO: Add your Prediction Model Code here
# ======================
if home_team and away_team and home_team != away_team:
    st.write(f"🔮 Prediction model will calculate outcome for **{home_team} vs {away_team}** ...")

