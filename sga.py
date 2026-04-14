import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import re

st.title("Skill Gap Analysis Tool")
st.subheader("Bridging the Gap Between University Curriculum and Industry Needs")

st.sidebar.header("Upload Required CSV Files")
uni_file = st.sidebar.file_uploader("University Curriculum (Topics)", type="csv")
ind_file = st.sidebar.file_uploader("Industry Requirements (Skills)", type="csv")
assess_file = st.sidebar.file_uploader("Assessments (Skills Assessed)", type="csv")

def preprocess_text(text_series):
    text = ' '.join(text_series.astype(str).tolist()).lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

if uni_file and ind_file and assess_file:
    uni_df = pd.read_csv(uni_file)
    ind_df = pd.read_csv(ind_file)
    assess_df = pd.read_csv(assess_file)

    # Preprocess all text
    uni_text = preprocess_text(uni_df['Topics'])
    ind_text = preprocess_text(ind_df['Skills'])
    assess_text = preprocess_text(assess_df['Skills_Assessed'])

    # Use a single vectorizer to build consistent vocabulary
    vectorizer = CountVectorizer()
    all_texts = [uni_text, ind_text, assess_text]
    matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Extract features for each
    uni_features = set(vectorizer.inverse_transform(matrix[0])[0])
    ind_features = set(vectorizer.inverse_transform(matrix[1])[0])
    assess_features = set(vectorizer.inverse_transform(matrix[2])[0])

    # Calculate gaps
    industry_gaps = sorted(ind_features - uni_features)
    assessment_gaps = sorted(assess_features - uni_features)

    # Display Results
    st.header("🔍 Skill Gap Results")

    st.subheader("📌 Industry Skill Gaps")
    st.write(f"🧩 Total: {len(industry_gaps)} missing skills from curriculum")
    st.write(industry_gaps)

    st.subheader("📌 Assessment Skill Gaps")
    st.write(f"🧪 Total: {len(assessment_gaps)} skills tested but not taught")
    st.write(assessment_gaps)

    # Visualization
    st.subheader("📊 Visual Summary")
    fig, ax = plt.subplots()
    categories = ['Industry Gaps', 'Assessment Gaps']
    values = [len(industry_gaps), len(assessment_gaps)]
    ax.bar(categories, values, color=['crimson', 'steelblue'])
    ax.set_ylabel('Number of Missing Skills')
    ax.set_title('Skill Gap Overview')
    st.pyplot(fig)

else:
    st.info("Please upload all three required CSV files to view the analysis.")
