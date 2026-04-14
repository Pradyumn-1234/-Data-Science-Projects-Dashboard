import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeRegressor

st.title("Language vs Placement Correlation")

file = st.file_uploader("Upload your CSV file", type='csv')
df = None 

if file is not None:
    try:

        # laod data
        df = pd.read_csv(file)
        st.subheader("Dataset Preview")
        st.dataframe(df, hide_index=True)

        # trim space
        df['Known_Languages'] = df['Known_Languages'].str.split(',').apply(lambda x: [lang.strip() for lang in x])



        # Encoding
        mlb = MultiLabelBinarizer()
        one_hot_encoded = mlb.fit_transform(df['Known_Languages'])
        lang = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)

        new_df = pd.concat([df.drop('Known_Languages', axis=1), lang], axis=1)



        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        data = pd.concat([new_df['Package_LPA'], lang], axis=1)
        corr_matrix = data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix[['Package_LPA']].sort_values('Package_LPA', ascending=False),
            annot=True,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax
        )
        st.pyplot(fig)



        # model 

        X = lang
        y = df['Package_LPA']
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X, y)


        # Most Important Languages
        st.subheader("Which Languages are Most Important for Placement/Salary?")

        importance = pd.Series(model.feature_importances_, index=X.columns)
        sorted_importance = importance.sort_values(ascending=False).reset_index()
        sorted_importance.columns = ['Language', 'Importance']
     
        st.dataframe(sorted_importance, hide_index=True)

        
        st.subheader("Visual Representation")
        st.bar_chart(data=sorted_importance.set_index('Language'))




        # Salary Prediction
        st.header("Predict Salary Based on Known Languages")
        dropdown = st.multiselect("Select Known Programming Languages:", options=X.columns, default=['Python'] )

        btn=st.button("Predict Salary!")

        if btn:
            input = [1 if lang in dropdown else 0 for lang in X.columns]
            
            salary = model.predict([input])[0]
            st.success(f"Estimated Salary: ₹ {salary:.2f} LPA")


    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")

else:
    st.info("Please upload a CSV file to preview and analyze the data.")
