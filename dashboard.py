import streamlit as st
import pandas as pd
import mysql.connector

# === MySQL Config ===
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',       # 🔐 Replace
    'password': '12345',   # 🔐 Replace
    'database': 'facial_access_db'   # ✅ Use your DB name
}

# === Fetch Data Function ===
def fetch_logs():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        query = "SELECT * FROM access_logs ORDER BY timestamp DESC"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except mysql.connector.Error as err:
        st.error(f"MySQL Error: {err}")
        return pd.DataFrame()

# === Streamlit UI ===
st.set_page_config(page_title="Secure Access Logs", layout="wide")
st.title("🔐 Facial Recognition Access Logs")

df = fetch_logs()

if df.empty:
    st.warning("No logs found or error connecting to MySQL.")
else:
    st.success(f"{len(df)} records loaded.")
    st.dataframe(df, use_container_width=True)

    # Optional filters
    with st.expander("🔍 Filter Logs"):
        col1, col2 = st.columns(2)
        with col1:
            selected_name = st.selectbox("Filter by Person", ["All"] + sorted(df['person_name'].unique().tolist()))
            if selected_name != "All":
                df = df[df['person_name'] == selected_name]

        with col2:
            selected_result = st.selectbox("Filter by Access Result", ["All", "Granted", "Denied"])
            if selected_result != "All":
                df = df[df['success'] == (1 if selected_result == "Granted" else 0)]

        st.dataframe(df, use_container_width=True)
