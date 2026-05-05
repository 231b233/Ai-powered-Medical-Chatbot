import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_text_model():
    print("Setting up the Text ML Model Training Pipeline...")
    
    # Path to your symptoms CSV file
    dataset_path = 'symptoms_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file '{dataset_path}' not found.")
        print("Please download a symptoms-disease dataset from Kaggle.")
        print("Example: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
        print("Save it as 'symptoms_dataset.csv' in this folder.")
        
        print("\nCreating a temporary dummy dataset so you can test the code...")
        dummy_data = {
            'Symptom_1': ['fever', 'cough', 'headache', 'stomach_pain', 'fever', 'headache', 'cough', 'vomiting', 'rash', 'dizziness'],
            'Symptom_2': ['cough', 'fever', 'dizziness', 'vomiting', 'rash', 'fever', 'headache', 'stomach_pain', 'fever', 'headache'],
            'Disease': ['Flu', 'Flu', 'Migraine', 'Food Poisoning', 'Chickenpox', 'Flu', 'Flu', 'Food Poisoning', 'Chickenpox', 'Migraine']
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(dataset_path, index=False)
        print("Dummy 'symptoms_dataset.csv' created!\n")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} records.")
    
    # Preprocessing: separate features (symptoms) from the target (disease)
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Convert text symptoms into numerical data that the algorithm can understand (One-Hot Encoding)
    X_encoded = pd.get_dummies(X)
    
    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Initialize the Machine Learning Algorithm (Random Forest is very accurate for text/symptoms)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    print("Training the Random Forest model on symptom data...")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, predictions)
    print("\n--- Final Text Model Evaluation ---")
    print(f"Final Testing Accuracy: {accuracy * 100:.2f}%\n")
    
    model_data = {
        'model': model,
        'columns': X_encoded.columns
    }
    joblib.save(model_data, 'text_disease_model.pkl')
    print("Model trained and saved as 'text_disease_model.pkl'!")

if __name__ == '__main__':
    train_text_model()
