import streamlit as st
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import os

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Employee Performance Dashboard", layout="wide")
st.title("Employee Performance Analysis")
st.markdown("""
This dashboard provides an interactive summary of employee performance data from INX, including key metrics, trends, and recommendations.
""")

# Load data
data_file = "Employee_Performance_CDS_Project_Data.xlsx"
df = pd.read_excel(data_file)

models = joblib.load("employee_performance_prediction_model.pkl")

# Sidebar filters
st.sidebar.header("Filter Data")
departments = df["EmpDepartment"].unique().tolist()
department = st.sidebar.selectbox("Select Department", ["All"] + departments)
if department != "All":
    df = df[df["EmpDepartment"] == department]
    

# Load the trained models
def load_model():
    try:
        if os.path.exists('employee_performance_prediction_model.pkl'):
            with open('employee_performance_prediction_model.pkl', 'rb') as file:
                models = pickle.load(file)
                if not hasattr(models, 'predict'):
                    st.error("Loaded model is not a valid classifier. Using placeholder model.")
                    models = create_placeholder_model()
        else:
            st.warning("Model file 'rf_model.pkl' not found. Using placeholder model.")
            models = create_placeholder_model()
    except Exception as e:
        st.error(f"Error loading model: {e}. Using placeholder model.")
        models = create_placeholder_model()
    return models

# Define categorical mappings based on the dataset
categorical_mappings = {
    'Gender': ['Male', 'Female'],
    'EducationBackground': ['Marketing', 'Life Sciences', 'Human Resources', 'Medical', 'Technical Degree', 'Other'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'EmpDepartment': ['Sales', 'Human Resources', 'Data Science', 'Development', 'Research & Development', 'Finance'],
    'EmpJobRole': ['Sales Executive', 'Manager', 'Developer', 'Data Scientist', 'Research Scientist', 'Finance Manager'],
    'BusinessTravelFrequency': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'OverTime': ['Yes', 'No'],
    'Attrition': ['Yes', 'No']
}

# Initialize LabelEncoders for categorical variables
label_encoders = {col: LabelEncoder().fit(values) for col, values in categorical_mappings.items()}

# Create input form in sidebar
with st.sidebar:
    st.header("Employee Details Input")
    
    # Numerical inputs
    age = st.number_input("Age", min_value=18, max_value=60, value=37, step=1)
    distance_from_home = st.number_input("Distance from Home (miles)", min_value=1, max_value=50, value=10, step=1)
    emp_education_level = st.selectbox("Education Level (1-5)", options=[1, 2, 3, 4, 5], index=2)
    emp_environment_satisfaction = st.selectbox("Environment Satisfaction (1-4)", options=[1, 2, 3, 4], index=2)
    emp_hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65, step=1)
    emp_job_involvement = st.selectbox("Job Involvement (1-4)", options=[1, 2, 3, 4], index=2)
    emp_job_level = st.selectbox("Job Level (1-5)", options=[1, 2, 3, 4, 5], index=2)
    emp_job_satisfaction = st.selectbox("Job Satisfaction (1-4)", options=[1, 2, 3, 4], index=2)
    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2, step=1)
    emp_last_salary_hike_percent = st.number_input("Last Salary Hike (%)", min_value=10, max_value=25, value=15, step=1)
    
    # Categorical inputs
    gender = st.selectbox("Gender", options=categorical_mappings['Gender'])
    education_background = st.selectbox("Education Background", options=categorical_mappings['EducationBackground'])
    marital_status = st.selectbox("Marital Status", options=categorical_mappings['MaritalStatus'])
    emp_department = st.selectbox("Department", options=categorical_mappings['EmpDepartment'])
    emp_job_role = st.selectbox("Job Role", options=categorical_mappings['EmpJobRole'])
    business_travel_frequency = st.selectbox("Business Travel Frequency", options=categorical_mappings['BusinessTravelFrequency'])
    over_time = st.selectbox("Over Time", options=categorical_mappings['OverTime'])
    emp_relationship_satisfaction = st.selectbox("Relationship Satisfaction (1-4)", options=[1, 2, 3, 4], index=2)
    total_work_experience = st.number_input("Total Work Experience (years)", min_value=0, max_value=40, value=7, step=1)
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2, step=1)
    emp_work_life_balance = st.selectbox("Work-Life Balance (1-4)", options=[1, 2, 3, 4], index=2)
    experience_years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=7, step=1)
    experience_years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=4, step=1)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=2, step=1)
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=4, step=1)
    attrition = st.selectbox("Attrition", options=categorical_mappings['Attrition'])

# Prepare input data
input_data = {
    'Age': age,
    'Gender': gender,
    'EducationBackground': education_background,
    'MaritalStatus': marital_status,
    'EmpDepartment': emp_department,
    'EmpJobRole': emp_job_role,
    'BusinessTravelFrequency': business_travel_frequency,
    'DistanceFromHome': distance_from_home,
    'EmpEducationLevel': emp_education_level,
    'EmpEnvironmentSatisfaction': emp_environment_satisfaction,
    'EmpHourlyRate': emp_hourly_rate,
    'EmpJobInvolvement': emp_job_involvement,
    'EmpJobLevel': emp_job_level,
    'EmpJobSatisfaction': emp_job_satisfaction,
    'NumCompaniesWorked': num_companies_worked,
    'OverTime': over_time,
    'EmpLastSalaryHikePercent': emp_last_salary_hike_percent,
    'EmpRelationshipSatisfaction': emp_relationship_satisfaction,
    'TotalWorkExperienceInYears': total_work_experience,
    'TrainingTimesLastYear': training_times_last_year,
    'EmpWorkLifeBalance': emp_work_life_balance,
    'ExperienceYearsAtThisCompany': experience_years_at_company,
    'ExperienceYearsInCurrentRole': experience_years_in_current_role,
    'YearsSinceLastPromotion': years_since_last_promotion,
    'YearsWithCurrManager': years_with_curr_manager,
    'Attrition': attrition
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical variables
for col in categorical_mappings.keys():
    if col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure all required features are present
required_features = [
    'Age', 'Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment',
    'EmpJobRole', 'BusinessTravelFrequency', 'DistanceFromHome', 'EmpEducationLevel',
    'EmpEnvironmentSatisfaction', 'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel',
    'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime', 'EmpLastSalaryHikePercent',
    'EmpRelationshipSatisfaction', 'TotalWorkExperienceInYears', 'TrainingTimesLastYear',
    'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Attrition'
]
input_df = input_df[required_features]

# Standardize numerical features
scaler = StandardScaler()
numerical_features = [
    'Age', 'DistanceFromHome', 'EmpHourlyRate', 'NumCompaniesWorked',
    'EmpLastSalaryHikePercent', 'TotalWorkExperienceInYears', 'TrainingTimesLastYear',
    'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
input_df[numerical_features] = scaler.fit_transform(input_df[numerical_features])

# Predict button in sidebar
with st.sidebar:
    if st.button("Predict Performance"):
        # Make prediction
        prediction = models.predict(input_df)[0]
        prediction_proba = models.predict_proba(input_df)[0]

        # Map prediction to readable format
        rating_map = {2: "Below Expectations", 3: "Meets Expectations", 4: "Exceeds Expectations"}
        predicted_rating = rating_map.get(prediction, "Unknown")

        # Display results in main content
        st.header("Prediction Results")
        st.success(f"Predicted Performance Rating: **{predicted_rating}** ({prediction})")
        st.write("Prediction Probabilities:")
        for rating, prob in zip([2, 3, 4], prediction_proba):
            st.write(f"Rating {rating} ({rating_map.get(rating)}): {prob:.2%}")

# Show data preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.write(df.describe())


# Department-wise performance
st.subheader("Department-wise Performance")
dep_perf = df.groupby("EmpDepartment")["PerformanceRating"].mean().sort_values(ascending=False)
st.bar_chart(dep_perf)

# Create comprehensive department performance summary
dept_performance = df.groupby('EmpDepartment').agg({
    'PerformanceRating': ['count', 'mean', 'std', 'min', 'max'],
    'EmpJobSatisfaction': ['mean', 'std'],
    'EmpEnvironmentSatisfaction': ['mean', 'std'],
    'EmpWorkLifeBalance': ['mean', 'std'],
    'EmpHourlyRate': ['mean', 'std'],
    'EmpLastSalaryHikePercent': ['mean', 'std'],
    'Age': 'mean',
    'ExperienceYearsAtThisCompany': 'mean',
    'Attrition': lambda x: (x == 'Yes').sum()
}).round(2)

# Flatten column names with better naming
dept_performance.columns = [
    'Employee_Count', 'Avg_Performance', 'Performance_Std', 'Min_Performance', 'Max_Performance',
    'Avg_Job_Satisfaction', 'Job_Satisfaction_Std',
    'Avg_Env_Satisfaction', 'Env_Satisfaction_Std', 
    'Avg_Work_Life_Balance', 'Work_Life_Balance_Std',
    'Avg_Hourly_Rate', 'Hourly_Rate_Std',
    'Avg_Salary_Hike', 'Salary_Hike_Std',
    'Avg_Age', 'Avg_Experience', 'Attrition_Count'
]

# Add calculated metrics
dept_performance['Attrition_Rate'] = (dept_performance['Attrition_Count'] / 
                                     dept_performance['Employee_Count'] * 100).round(2)
dept_performance['Performance_Range'] = (dept_performance['Max_Performance'] - 
                                        dept_performance['Min_Performance']).round(2)
# Department Performance with Error Bars
fig, ax = plt.subplots(figsize=(12, 8))
dept_perf_data = dept_performance.reset_index()
bars = ax.bar(dept_perf_data['EmpDepartment'], dept_perf_data['Avg_Performance'], 
             yerr=dept_perf_data['Performance_Std'], capsize=5, 
             color='skyblue', alpha=0.8, edgecolor='black')
ax.set_title('Average Performance Rating by Department\n(with Standard Deviation)', 
           fontsize=16, fontweight='bold')
ax.set_ylabel('Average Performance Rating')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 4)

# Add value labels on bars
for bar, avg, std in zip(bars, dept_perf_data['Avg_Performance'], dept_perf_data['Performance_Std']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
          f'{avg:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
st.pyplot(fig)

# Prepare data for feature importance analysis
df_analysis = df.copy()

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
                      'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

for col in categorical_columns:
    le = LabelEncoder()
    df_analysis[col + '_encoded'] = le.fit_transform(df_analysis[col])
    label_encoders[col] = le

# Select features for analysis (excluding employee number and original categorical columns)
feature_columns = [col for col in df_analysis.columns if col not in 
                  ['EmpNumber', 'PerformanceRating'] + categorical_columns]

X = df_analysis[feature_columns]
y = df_analysis['PerformanceRating']

print("=== CORRELATION ANALYSIS ===")
# Calculate correlation with performance rating
correlations = df_analysis[feature_columns + ['PerformanceRating']].corr()['PerformanceRating'].abs().sort_values(ascending=False)
correlations = correlations.drop('PerformanceRating')  # Remove self-correlation

print("Top 10 Features by Correlation with Performance Rating:")
print(correlations.head(10))

# Feature Importance using Random Forest
print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

st.write("Top 10 Features by Random Forest Importance:")
st.write(feature_importance.head(10))

# Visualize Top Factors
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Top 10 Feature Importance
top_features = feature_importance.head(10)
top_features.plot(x='feature', y='importance', kind='barh', ax=axes[0,0], color='lightcoral')
axes[0,0].set_title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Importance Score')

# 2. Top 10 Correlations
top_corr = correlations.head(10)
top_corr.plot(kind='barh', ax=axes[0,1], color='lightblue')
axes[0,1].set_title('Top 10 Correlations with Performance Rating', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Absolute Correlation')

# 3. Top 3 Factors Analysis - Environment Satisfaction
df.boxplot(column='EmpEnvironmentSatisfaction', by='PerformanceRating', ax=axes[1,0])
axes[1,0].set_title('Environment Satisfaction vs Performance Rating', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Performance Rating')
axes[1,0].set_ylabel('Environment Satisfaction')
plt.setp(axes[1,0], title='')

# 4. Top 3 Factors Analysis - Salary Hike Percentage
df.boxplot(column='EmpLastSalaryHikePercent', by='PerformanceRating', ax=axes[1,1])
axes[1,1].set_title('Salary Hike % vs Performance Rating', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel('Performance Rating')
axes[1,1].set_ylabel('Last Salary Hike %')
plt.setp(axes[1,1], title='')

plt.tight_layout()
st.pyplot(fig)

# Identify and analyze TOP 3 FACTORS
top_3_factors = feature_importance.head(3)
for i, (_, row) in enumerate(top_3_factors.iterrows(), 1):
    factor_name = row['feature']
    importance = row['importance']
    
    print(f"\n{i}. {factor_name.replace('_', ' ').title()}")
    print(f"   Importance Score: {importance:.3f}")
    
    # Get correlation info
    correlation = correlations[factor_name] if factor_name in correlations else 0
    print(f"   Correlation with Performance: {correlation:.3f}")
    
    # Statistical analysis by performance rating
    if factor_name in df.columns:
        factor_analysis = df.groupby('PerformanceRating')[factor_name].agg(['mean', 'std']).round(2)
        print(f"   Performance Rating Analysis:")
        for rating in sorted(df['PerformanceRating'].unique()):
            mean_val = factor_analysis.loc[rating, 'mean']
            print(f"     Rating {rating}: Mean = {mean_val}")

# Display top 3 factors
st.subheader("Top 3 Factors Affecting Performance")
st.write("1. 🌟 EMPLOYEE ENVIRONMENT SATISFACTION: Most correlated with performance")
st.write("2. 💰 SALARY HIKE PERCENTAGE: Strongest feature importance predictor") 
st.write("3. ⏰ YEARS SINCE LAST PROMOTION: Third most important factor")


# Recommendations
st.subheader("Recommendations")
st.markdown("""
- Increase training opportunities for underperforming departments.
- Monitor absenteeism and address root causes.
- Implement targeted retention strategies.
""")

st.markdown("---")
st.markdown("**Author:** Emma Kawira")
