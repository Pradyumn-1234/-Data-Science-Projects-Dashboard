import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Bussiness Forecast System")

file = st.file_uploader("Upload Excel File:", type=['xlsx'])

if file is not None:
    try:
        # Load the Excel file
        df = pd.read_excel(file)
        st.subheader("Raw Data")
        st.dataframe(df)

        # Date conversion
        df['Reg.Date'] = pd.to_datetime(df['Reg.Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Reg.Date'])  # Remove rows with invalid dates
        df['Month_Year'] = df['Reg.Date'].dt.to_period('M')

        # Filters
        tech = st.selectbox("Technology", sorted(df['Subject'].dropna().unique()))
        college = st.selectbox("Filter by College (Optional)", ['All'] + sorted(df['College'].fillna('Not Provided').unique()))
        location = st.selectbox("Filter by Location (Optional)", ['All'] + sorted(df['Location'].dropna().unique()))

        # Apply filters
        filtered_df = df[df['Subject'] == tech]
        if college != 'All':
            filtered_df = filtered_df[filtered_df['College'].fillna('Not Provided') == college]
        if location != 'All':
            filtered_df = filtered_df[filtered_df['Location'] == location]

        # Monthly aggregation
        enrollments = filtered_df.groupby('Month_Year').size().reset_index(name='Count')

        if len(enrollments) >= 2:
            enrollments['Ordinal'] = enrollments['Month_Year'].dt.to_timestamp().map(lambda x: x.toordinal())

            X = enrollments[['Ordinal']]
            y = enrollments['Count']

            model = LinearRegression()
            model.fit(X, y)

            # Predict for future 6 months
            last_date = enrollments['Month_Year'].max().to_timestamp()
            future_months = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=6, freq='MS')
            future_ordinals = future_months.map(lambda x: x.toordinal())

            future_preds = model.predict(future_ordinals.values.reshape(-1, 1))

            future_df = pd.DataFrame({
                'Month_Year': future_months.to_period('M'),
                'Predicted_Count': future_preds.astype(int)
            })

            # Combine actual + predicted for chart
            chart_df = pd.concat([
                enrollments[['Month_Year', 'Count']].rename(columns={'Count': 'Enrollments'}),
                future_df.rename(columns={'Predicted_Count': 'Enrollments'})
            ], ignore_index=True)

            chart_df = chart_df.sort_values('Month_Year')
            chart_df['Month_Year'] = chart_df['Month_Year'].astype(str)

            st.subheader("📊 Enrollment Forecast")
            st.line_chart(data=chart_df.set_index('Month_Year'))

            st.subheader("📋 Predicted Enrollments (Next 6 Months)")
            st.dataframe(future_df)

        else:
            st.warning("Not enough data points to train model. Minimum 2 months required.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("Please upload a .xlsx file to proceed.")
