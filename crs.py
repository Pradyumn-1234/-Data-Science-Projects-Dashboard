import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Title
st.title("🎯 SPI Course Recommender")

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv('spi_data.csv')
    df.columns=df.columns.str.strip().str.replace(" ","_").str.replace("-","_") 
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()


col = st.selectbox("College", sorted(df['College'].unique()), index=None, placeholder="-- Select Your College --")
corse = st.selectbox("Course", sorted(df['Course'].unique()), index=None, placeholder="-- Select Your Course --")
branch = st.selectbox("Branch", sorted(df['Branch'].unique()), index=None, placeholder="-- Select Your Branch --")
year = st.selectbox("Year", sorted(df['Year'].unique()), index=None, placeholder="-- Select Your Year --")


# Train model 
@st.cache_data
def train_model(df):

    # Encoding
    le_college = LabelEncoder()
    le_course = LabelEncoder()
    le_branch = LabelEncoder()
    le_subject = LabelEncoder()
    
    df['College'] = le_college.fit_transform(df['College'])
    df['Course'] = le_course.fit_transform(df['Course'])
    df['Branch'] = le_branch.fit_transform(df['Branch'])
    df['Subject'] = le_subject.fit_transform(df['Subject'])
    

    # Train-test split
    X = df[['College', 'Course', 'Branch', 'Year']]
    y = df['Subject']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Scaling
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    

    # Model training
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    
    return rfc, le_college, le_course, le_branch, le_subject, ss

rfc, le_college, le_course, le_branch, le_subject, ss = train_model(df)



# prediction
def predict_subject(col, corse, branch, year):
    
    if None in [col, corse, branch, year]:
        st.warning("⚠️ Please select all fields!")
        return None
    
    try:
       
        new_df = pd.DataFrame({
            'College': [col],
            'Course': [corse],
            'Branch': [branch],
            'Year': [year]
        })
        
        # Transform inputs
        new_df['College'] = le_college.transform(new_df['College'])
        new_df['Course'] = le_course.transform(new_df['Course'])
        new_df['Branch'] = le_branch.transform(new_df['Branch'])
        
        # Scale and predict
        new_df_scaled = ss.transform(new_df)
        prediction = rfc.predict(new_df_scaled)
        return le_subject.inverse_transform(prediction)[0]
    
    except (ValueError, NotFittedError) as e:
        st.error(f"❌ Error: {str(e)} (Did you select a valid option?)")
        return None

# Predict button
if st.button("🔍 Recommended Subjects (ML-Based)"):
    result = predict_subject(col, corse, branch, year)
    if result:
        st.success(f"🎓 Recommended Subject: **{result}**")