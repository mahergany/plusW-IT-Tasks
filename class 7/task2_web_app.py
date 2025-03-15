import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('salary_data.csv')  # Ensure this file is in the same directory
df.dropna(inplace=True)
if 'YearsExperience' not in df.columns or 'Salary' not in df.columns:
    st.error("Error: The dataset must contain 'YearsExperience' and 'Salary' columns.")
    st.stop()

X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

#streamlit
st.title("Salary Prediction System")
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0)

input_data = pd.DataFrame({'YearsExperience': [experience]})
predicted_salary = model.predict(input_data)[0]

st.write(f"### Predicted Salary: ${predicted_salary:,.2f}")
st.write("#### Model Performance")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- R-squared: {r2:.2f}")
