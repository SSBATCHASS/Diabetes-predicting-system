import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the Pima Indians Diabetes Dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=column_names)
    return data

def explore_data(df):
    """Explore and visualize the dataset"""
    # Basic information
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values (zeros in certain columns are considered missing)
    print("\nMissing Values (zeros):")
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_columns:
        zero_count = (df[column] == 0).sum()
        print(f"{column}: {zero_count} zeros ({zero_count/len(df)*100:.2f}%)")
    
    # Class distribution
    print("\nClass Distribution:")
    class_counts = df['Outcome'].value_counts()
    print(class_counts)
    print(f"Percentage of diabetic patients: {class_counts[1]/len(df)*100:.2f}%")
    
    # Create visualizations directory
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Visualizations
    
    # 1. Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df, palette='Set1')
    plt.title('Class Distribution (0: No Diabetes, 1: Diabetes)')
    plt.savefig('visualizations/class_distribution.png')
    
    # 2. Feature distributions by class
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns[:-1]):
        plt.subplot(2, 4, i+1)
        sns.histplot(data=df, x=column, hue='Outcome', kde=True, palette='Set1')
        plt.title(f'{column} Distribution by Class')
    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    
    # 3. Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('visualizations/correlation_matrix.png')
    
    # 4. Box plots for each feature
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns[:-1]):
        plt.subplot(2, 4, i+1)
        sns.boxplot(x='Outcome', y=column, data=df, palette='Set1')
        plt.title(f'{column} by Diabetes Outcome')
    plt.tight_layout()
    plt.savefig('visualizations/boxplots.png')
    
    # 5. Pairplot for key features
    key_features = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[key_features], hue='Outcome', palette='Set1')
    plt.savefig('visualizations/pairplot.png')
    
    print("\nData exploration completed. Visualizations saved in 'visualizations' directory.")

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Explore data
    print("\nExploring data...")
    explore_data(data)