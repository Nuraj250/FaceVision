import streamlit as st
import pandas as pd
import time
import os

LOG_FILE = "logs/recognitions.csv"

st.set_page_config(page_title="FaceVision Dashboard", layout="wide")

st.title("📸 FaceVision Live Dashboard")
st.caption("Real-time facial recognition logs with attendance count")

placeholder = st.empty()

def load_logs():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df.sort_values("Timestamp", ascending=False)
    return pd.DataFrame(columns=["Timestamp", "Name", "Confidence (%)", "Thumbnail"])

def count_attendance(df):
    return df['Name'].value_counts().reset_index().rename(
        columns={"index": "Name", "Name": "Recognitions"}
    )

while True:
    df = load_logs()

    with placeholder.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Recognition Log")
            st.dataframe(df.head(15), use_container_width=True)

        with col2:
            st.subheader("👥 Attendance Count")
            count_df = count_attendance(df)
            st.bar_chart(count_df.set_index("Name"))

    time.sleep(5)
