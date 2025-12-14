# Diabetes Prediction Project

This is a machine learning project that predicts the likelihood of diabetes based on various health metrics. The project uses the Pima Indians Diabetes Dataset and implements a Random Forest classifier for prediction.

## Features

- Data preprocessing and exploration
- Machine learning model training with hyperparameter tuning
- Interactive web interface for making predictions
- Model evaluation metrics and visualizations

## Dataset

The Pima Indians Diabetes Dataset includes the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (2 hours after an oral glucose tolerance test)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0 or 1) indicating diabetes diagnosis

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model separately:
```
python model.py
```

This will:
- Load and preprocess the data
- Train a Random Forest model with hyperparameter tuning
- Evaluate the model and save it to disk

### Running the Web Application

To run the interactive web application:
```
streamlit run app.py
```

This will:
- Load the dataset
- Train the model (if not already trained)
- Launch a web interface where you can input health metrics and get a diabetes risk prediction

## Project Structure

- `app.py`: Streamlit web application
- `model.py`: Script for training and evaluating the model
- `requirements.txt`: List of required Python packages
- `diabetes_model.pkl`: Saved model file (generated after training)
- `feature_importance.png`: Feature importance visualization (generated after training)

## Model Performance

The Random Forest model achieves approximately 75-80% accuracy on the test set. The most important features for prediction are typically Glucose, BMI, and Age.

## Future Improvements

- Implement additional machine learning algorithms for comparison
- Add more advanced feature engineering techniques
- Enhance the user interface with additional visualizations
- Deploy the application to a cloud platform for wider access