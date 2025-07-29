import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AfriDSCA Employee Performance Dashboard", layout="wide")
st.title("AfriDSCA Employee Performance Analysis")
st.markdown("""
This dashboard provides an interactive summary of employee performance data from INX, including key metrics, trends, and recommendations.
""")

# Load data
data_file = "Employee_Performance_CDS_Project_Data.xlsx"
df = pd.read_excel(data_file)

# Sidebar filters
st.sidebar.header("Filter Data")
departments = df["Department"].unique().tolist()
department = st.sidebar.selectbox("Select Department", ["All"] + departments)
if department != "All":
    df = df[df["Department"] == department]

# Show data preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Performance rating distribution
st.subheader("Performance Rating Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Performance Rating"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Department-wise performance
st.subheader("Department-wise Performance")
dep_perf = df.groupby("Department")["Performance Rating"].mean().sort_values(ascending=False)
st.bar_chart(dep_perf)

# Training hours vs performance
st.subheader("Training Hours vs Performance")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="Training Hours", y="Performance Rating", data=df, ax=ax2)
st.pyplot(fig2)

# Absenteeism analysis
st.subheader("Absenteeism Analysis")
absent_rate = df.groupby("Department")["Absenteeism"].mean()
st.bar_chart(absent_rate)

# Recommendations
st.subheader("Recommendations")
st.markdown("""
- Increase training opportunities for underperforming departments.
- Monitor absenteeism and address root causes.
- Implement targeted retention strategies.
""")

st.markdown("---")
st.markdown("**Author:** Emma Kawira")
