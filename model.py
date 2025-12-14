import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the Pima Indians Diabetes Dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=column_names)
    return data

def preprocess_data(df):
    """Preprocess the data by handling missing values and scaling features"""
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Replace zeros with NaN in specific columns
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        df_processed[column] = df_processed[column].replace(0, np.NaN)
    
    # Fill NaN values with median
    for column in zero_columns:
        df_processed[column] = df_processed[column].fillna(df_processed[column].median())
    
    # Split features and target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a Random Forest model with hyperparameter tuning"""
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                              cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    return best_model, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm, y_pred

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    # Get feature importance
    importances = model.feature_importances_
    
    # Sort feature importance in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [feature_names[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), names, rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')

def save_model(model, scaler, filename='diabetes_model.pkl'):
    """Save the trained model and scaler to disk"""
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, filename)
    print(f"Model saved as {filename}")

def main():
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Display basic information
    print("\nDataset Information:")
    print(f"Shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Train model
    print("\nTraining model...")
    model, best_params = train_model(X_train, y_train)
    print(f"\nBest parameters: {best_params}")
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, report, cm, y_pred = evaluate_model(model, X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(model, data.columns[:-1])
    
    # Save model
    print("\nSaving model...")
    save_model(model, scaler)
    
    print("\nModel training and evaluation completed!")

if __name__ == "__main__":
    main()