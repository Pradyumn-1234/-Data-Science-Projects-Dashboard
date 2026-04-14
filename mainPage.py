import streamlit as st
import subprocess

st.header(" :rocket: Data Science Projects Dashboard")

col1, col2 = st.columns(2, gap = "small", vertical_alignment="center")

with col1:
    if st.button(" :books: Course Recommender System"):
        subprocess.run(["streamlit", "run", "crs.py"])
        
        
    if st.button(" :chart_with_upwards_trend: Language Vs. Placement Correlation"):
        subprocess.run(["streamlit", "run", "lps.py"])
        
with col2:
    if st.button(" :cityscape: Business Forecast System"):
        subprocess.run(["streamlit", "run", "bfs.py"])

    if st.button(" :sports_medal: Skill Gap Analysis"):
        subprocess.run(["streamlit", "run", "sga.py"])

if st.button(" :weight_lifter: Student Performance Prediction"):
    subprocess.run(["streamlit", "run", "gp.py"])

st.badge("Make sure CRS,BFS, LVPC, SGA, and SPP are installed in your system")