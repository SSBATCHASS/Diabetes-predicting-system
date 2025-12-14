import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Function to load data
@st.cache_data
def load_data():
    # Using the Pima Indians Diabetes Database
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=column_names)
    return data

# Function to preprocess data
def preprocess_data(df):
    # Replace zeros with NaN in specific columns
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        df[column] = df[column].replace(0, np.NaN)
    
    # Fill NaN values with median
    for column in zero_columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

# Function to train model
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X_test, y_test, y_pred

# Function to make prediction
def predict_diabetes(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    return prediction[0], probability[0][1]

# Main function
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Info"])
    
    # Load data
    data = load_data()
    X, y = preprocess_data(data)
    
    # Train model if not already trained
    if 'model' not in st.session_state:
        with st.spinner("Training model..."):
            model, scaler, accuracy, X_test, y_test, y_pred = train_model(X, y)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.accuracy = accuracy
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
    
    # Home page
    if page == "Home":
        st.title("Diabetes Prediction App")
        st.write("Welcome to the Diabetes Prediction App! This application uses machine learning to predict the likelihood of diabetes based on several health metrics.")
        
        st.subheader("Dataset Overview")
        st.write("This app uses the Pima Indians Diabetes Database, which includes the following features:")
        st.write(pd.DataFrame({
            'Feature': data.columns[:-1],
            'Description': [
                'Number of pregnancies',
                'Plasma glucose concentration (mg/dL)',
                'Diastolic blood pressure (mm Hg)',
                'Triceps skinfold thickness (mm)',
                'Insulin level (mu U/ml)',
                'Body mass index (weight in kg/(height in m)²)',
                'Diabetes pedigree function',
                'Age (years)'
            ]
        }))
        
        st.subheader("Data Sample")
        st.dataframe(data.head())
        
    # Prediction page
    elif page == "Prediction":
        st.title("Diabetes Prediction")
        st.write("Enter your health metrics below to get a diabetes risk prediction.")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
        # Make prediction when button is clicked
        if st.button("Predict"):
            features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            prediction, probability = predict_diabetes(st.session_state.model, st.session_state.scaler, features)
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"High risk of diabetes (Probability: {probability:.2f})")
            else:
                st.success(f"Low risk of diabetes (Probability: {probability:.2f})")
            
            # Display gauge chart for probability
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "green"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            st.plotly_chart(fig)
    
    # Model Info page
    elif page == "Model Info":
        st.title("Model Information")
        st.write(f"Model Accuracy: {st.session_state.accuracy:.2f}")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=['Actual Negative', 'Actual Positive'], 
                            columns=['Predicted Negative', 'Predicted Positive'])
        st.dataframe(cm_df)
        
        st.subheader("Classification Report")
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(feature_importance.set_index('Feature'))

if __name__ == "__main__":
    main()