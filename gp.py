import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load csv
df = pd.read_csv("softpro_student_data.csv")
df = df.drop(columns=['Enrollment_no','Name','Presentation','Attendance'])


# title
st.title("🎓 Student Grade Prediction App")
st.markdown("Fill the student's score to predict their final grade")
st.markdown("  ")


numeric_cols = [col for col in df.select_dtypes(include='number').columns if col != 'Technology_encoded']

# inputs
tech_input = st.selectbox("📚 Technology", sorted(df["Technology"].unique()), index=None, placeholder="Select your Technology")

selected_values = {}
for col in numeric_cols:

    min_val = int(df[col].min())  
    max_val = int(df[col].max())  
    default_val = int(df[col].mean()) 
    selected_values[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=default_val )



# encoding of categorical column
le=LabelEncoder()
df['Technology_encoded']=le.fit_transform(df['Technology'])


# features and target
X = df.drop(['Grade', 'Technology'], axis=1)
y = df['Grade']


# train-test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# scaling
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

# model
rfc=RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)
rfc.fit(X_train,y_train)


# button
if st.button("🎯 Predict Grade!"):
    if tech_input is None:
        st.warning("⚠️ Please select a Technology first.")
    else:
        # conveting cat into num
        tech_encoded = le.transform([tech_input])[0]
        
        input_features = [tech_encoded] + [selected_values[col] for col in numeric_cols]
        input_array = np.array(input_features).reshape(1, -1)
        
        # scaling
        scaled_input = ss.transform(input_array)
        
        # Get grade prediction from model
        prediction = rfc.predict(scaled_input)[0]
        st.success(f"📢 Predicted Grade: {prediction}")