import streamlit as st 
import pandas as pd


st.title("only for practice")

file=st.file_uploader("upload excel file:", type=[xlsx])

if file is not None:
    try:
        #load data frame 
        df=pd.read_excel("")
        
    except Exception as e:
        st.error(f"error",{str(e)})

else:
    st.info("please upload xlsx file again")