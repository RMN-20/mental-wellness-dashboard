import streamlit as st
import pandas as pd
from datetime import datetime
import os
from sentiment import analyze_sentiment, analyze_journal_csv, classify_mood
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Mental Wellness Sentiment Dashboard", layout="centered")
st.title("Mental Wellness Sentiment Dashboard")

data_path = "data/sample_data.csv"

st.subheader("New Journal Entry")

with st.form("entry_form"):
    entry_text = st.text_area("How are you feeling today?", height=150)
    submitted = st.form_submit_button("Submit Entry")

if submitted and entry_text.strip() != "":
    today = datetime.today().strftime("%Y-%m-%d")
    sentiment_scores = analyze_sentiment(entry_text)
    mood = classify_mood(sentiment_scores)

    new_row = {
        "date": today,
        "entry": entry_text,
        "TextBlob Polarity": sentiment_scores["TextBlob Polarity"],
        "VADER Compound": sentiment_scores["VADER Compound"],
        "VADER Positive": sentiment_scores["VADER Positive"],
        "VADER Negative": sentiment_scores["VADER Negative"],
        "VADER Neutral": sentiment_scores["VADER Neutral"],
        "Mood": mood
    }

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df._append(new_row, ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(data_path, index=False)
    st.success("Entry added successfully!")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)

    for idx, row in df.iterrows():
        if pd.isnull(row.get("TextBlob Polarity")) or pd.isnull(row.get("Mood")):
            scores = analyze_sentiment(row["entry"])
            df.at[idx, "TextBlob Polarity"] = scores["TextBlob Polarity"]
            df.at[idx, "VADER Compound"] = scores["VADER Compound"]
            df.at[idx, "VADER Positive"] = scores["VADER Positive"]
            df.at[idx, "VADER Negative"] = scores["VADER Negative"]
            df.at[idx, "VADER Neutral"] = scores["VADER Neutral"]
            df.at[idx, "Mood"] = classify_mood(scores)

    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(data_path, index=False)
else:
    st.warning("No journal entries yet.")
    st.stop()

st.subheader("📑 Journal Data")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

st.subheader("Sentiment Summary")
avg_blob = df["TextBlob Polarity"].mean()
avg_vader = df["VADER Compound"].mean()

col1, col2 = st.columns(2)
col1.metric("TextBlob Avg Polarity", f"{avg_blob:.2f}")
col2.metric("VADER Avg Compound", f"{avg_vader:.2f}")

st.subheader("Sentiment Over Time")
fig = px.line(
    df,
    x="date",
    y=["TextBlob Polarity", "VADER Compound"],
    title="Sentiment Trend",
    labels={"value": "Sentiment Score", "date": "Date", "variable": "Method"}
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Mood Frequency")
mood_freq = df["Mood"].value_counts().reset_index()
mood_freq.columns = ["Mood", "Count"]
bar_fig = px.bar(mood_freq, x="Mood", y="Count", color="Mood", title="Mood Frequency")
st.plotly_chart(bar_fig, use_container_width=True)

st.subheader("Sentiment Trend with Mood Highlight")

df_sorted = df.sort_values("date")

import plotly.graph_objects as go

line_mood_fig = go.Figure()

line_mood_fig.add_trace(go.Scatter(
    x=df_sorted["date"],
    y=df_sorted["VADER Compound"],
    mode='lines',
    name='VADER Compound',
    line=dict(color='gray'),
))

line_mood_fig.add_trace(go.Scatter(
    x=df_sorted["date"],
    y=df_sorted["VADER Compound"],
    mode='markers',
    marker=dict(
        color=df_sorted["Mood"].map({
            'happy': '#1f77b4',
            'anxious': '#ff7f0e',
            'neutral': '#2ca02c'
        }),
        size=10
    ),
    name='Mood',
    text=df_sorted["Mood"]
))

line_mood_fig.update_layout(
    title="Sentiment Trend with Mood Classification",
    xaxis_title="Date",
    yaxis_title="VADER Compound Sentiment",
    showlegend=False
)

st.plotly_chart(line_mood_fig, use_container_width=True)

st.subheader("Overall Mood Distribution")
pie_fig = px.pie(df, names="Mood", title="Overall Mood Breakdown")
st.plotly_chart(pie_fig, use_container_width=True)
