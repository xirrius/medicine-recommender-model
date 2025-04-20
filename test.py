import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

def load_datasets():
    crop_file = 'crop_medicine_dataset.csv'
    animal_file = 'animal_medicine_dataset.csv'
    
    if os.path.exists(crop_file) and os.path.exists(animal_file):
        try:
            crop_df = pd.read_csv(crop_file)
            animal_df = pd.read_csv(animal_file)
            print(f"Successfully loaded datasets: {len(crop_df)} crop records, {len(animal_df)} animal records")
            return crop_df, animal_df
        except Exception as e:
            print(f"Error loading CSV files: {e}")
    
    print("Dataset files not found. Generating sample data...")
    from generate_datasets import generate_crop_dataset, generate_animal_dataset
    crop_df = generate_crop_dataset(500)
    animal_df = generate_animal_dataset(500)
    crop_df.to_csv(crop_file, index=False)
    animal_df.to_csv(animal_file, index=False)
    print("Generated and saved datasets: crop_medicine_dataset.csv, animal_medicine_dataset.csv")
    return crop_df, animal_df

def train_disease_predictor(data, type_name):
    features = ['region', 'crop' if type_name == 'crop' else 'animal', 
                'season' if type_name == 'crop' else 'age_group', 'symptom']
    X = data[features]
    y = data['disease']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42))
    ])
    
    model.fit(X, y)
    
    with open(f"{type_name}_disease_predictor.pkl", 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)
    
    return model, features

def engineer_features(df, data_type="crop", prediction_mode=False):

    df = df.copy()
    
    df['symptom_length'] = df['symptom'].str.len().fillna(0)
    
    if data_type == "crop":
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        for col in ['season_Kharif', 'season_Rabi']:
            if col not in season_dummies.columns:
                season_dummies[col] = 0
        df = pd.concat([df, season_dummies], axis=1)
        
    else:
        age_mapping = {
            'Baby': 0, 'Calf': 0, 'Kid': 0, 'Lamb': 0, 'Chick': 0, 'Puppy': 0,
            'Young': 1, 'Heifer': 1, 'Grower': 1,
            'Adult': 2, 'Layer': 2, 'Broiler': 2,
            'Senior': 3
        }
        df['age_numeric'] = df['age_group'].map(lambda x: next((v for k, v in age_mapping.items() if k in str(x)), 1))
    
    if prediction_mode:
        df['region_disease_count'] = 1
        df['disease_medicine_frequency'] = 1
    else:
        region_disease_counts = df.groupby(['region', 'disease']).size().reset_index(name='region_disease_count')
        df = df.merge(region_disease_counts, on=['region', 'disease'], how='left')
        disease_medicine_counts = df.groupby(['disease', 'recommended_medicine']).size().reset_index(name='disease_medicine_frequency')
        df = df.merge(disease_medicine_counts, on=['disease', 'recommended_medicine'], how='left')
        df['region_disease_count'] = df['region_disease_count'].fillna(1)
        df['disease_medicine_frequency'] = df['disease_medicine_frequency'].fillna(1)
    
    common_symptoms = ['yellow', 'wilt', 'spot', 'rot', 'fever', 'diarrhea', 'vomit', 'lame']
    for symptom in common_symptoms:
        df[f'symptom_{symptom}'] = df['symptom'].str.lower().str.contains(symptom, na=False).astype(int)
    
    return df

def build_model(data, type_name="crop"):
    print(f"\nBuilding {type_name} model...")
    
    data = engineer_features(data, type_name)
    
    if type_name == "crop":
        categorical_features = ['region', 'crop', 'season', 'disease', 'symptom']
        numeric_features = ['symptom_length', 'region_disease_count', 'disease_medicine_frequency']
        binary_features = [col for col in data.columns if col.startswith('symptom_')]
        binary_features += ['season_Kharif', 'season_Rabi']
    else:
        categorical_features = ['region', 'animal', 'disease', 'symptom']
        numeric_features = ['age_numeric', 'symptom_length', 'region_disease_count', 'disease_medicine_frequency']
        binary_features = [col for col in data.columns if col.startswith('symptom_')]
    
    # Ensure unique features
    features = categorical_features + numeric_features + binary_features
    features = list(dict.fromkeys(features))  # Remove duplicates
    print(f"Using features: {features}")
    
    X = data[features]
    y = data['recommended_medicine']
    
    class_counts = y.value_counts()
    print(f"Target class distribution (top 5):\n{class_counts.head()}")
    print(f"Total classes: {len(class_counts)}")
    min_class_count = class_counts.min()
    print(f"Smallest class count: {min_class_count}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features),
            ('bin', 'passthrough', binary_features)
        ])
    
    if min_class_count < 20:
        print("Applying SMOTE to balance classes...")
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        
        min_train_class_count = y_train.value_counts().min()
        print(f"Smallest class count in training data: {min_train_class_count}")
        
        k_neighbors = max(1, min(5, min_train_class_count - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
        except ValueError as e:
            print(f"SMOTE failed: {e}. Falling back to original data...")
            X_train_resampled, y_train_resampled = X_train_preprocessed, y_train
    else:
        X_train_resampled = preprocessor.fit_transform(X_train)
        y_train_resampled = y_train
    
    X_test_preprocessed = preprocessor.transform(X_test)
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    print(f"Training {type_name} model...")
    model.fit(X_train_resampled, y_train_resampled)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{type_name.capitalize()} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, features

def predict_medicine(input_data, crop_model_file, animal_model_file):
    logger.debug("Starting predict_medicine with input: %s", input_data)
    data_type = 'animal' if 'animal' in input_data else 'crop'
    model_file = animal_model_file if data_type == 'animal' else crop_model_file
    
    with open(model_file, 'rb') as file:
        model_data = pickle.load(file)
    
    model = model_data['model']
    features = model_data['features']
    
    # print("Input data received:", input_data)
    
    input_df = pd.DataFrame([input_data])
    # print("Initial input_df:", input_df.to_dict())
    
    if 'disease' not in input_df.columns or pd.isna(input_df['disease'].iloc[0]):
        
        print("Predicting disease...")
        
        disease_model_file = f"{data_type}_disease_predictor.pkl"
        with open(disease_model_file, 'rb') as f:
            disease_data = pickle.load(f)
        disease_model = disease_data['model']
        disease_features = disease_data['features']
        
        available_features = [f for f in disease_features if f in input_df.columns]
        if not available_features:
            raise ValueError("No features available for disease prediction")
        input_df['disease'] = disease_model.predict(input_df[available_features])[0]
    
    input_df = engineer_features(input_df, data_type, prediction_mode=True)
    
    input_features = {}
    for feature in features:
        if feature in input_df.columns:
            input_features[feature] = input_df[feature].values[0]
        else:
            if feature in ['symptom_length', 'age_numeric', 'region_disease_count', 'disease_medicine_frequency'] or feature.startswith('symptom_') or feature.startswith('season_'):
                input_features[feature] = 0
            else:
                input_features[feature] = 'unknown'
    
    prediction_input = pd.DataFrame([input_features])
    
    print("prediction_input:", prediction_input.to_dict())  # Log final input
    print("NaN check in prediction_input:", prediction_input.isna().sum().to_dict())  # Log NaN counts
    

    medicine = model.predict(prediction_input)[0]
    
    confidence = 0.75
    if hasattr(model.named_steps['classifier'], 'predict_proba'):
        try:
            probs = model.predict_proba(prediction_input)[0]
            class_idx = np.where(model.named_steps['classifier'].classes_ == medicine)[0][0]
            confidence = float(probs[class_idx])
        except:
            pass
    
    predicted_disease = input_df['disease'].values[0] if 'disease' not in input_data or pd.isna(input_data.get('disease')) else None
    
    return {
        'type': data_type,
        'recommendation': str(medicine),
        'confidence': confidence,
        'predicted_disease': str(predicted_disease) if predicted_disease is not None else None
    }

def main():
    crop_df, animal_df = load_datasets()
    
    crop_disease_model, crop_disease_features = train_disease_predictor(crop_df, "crop")
    animal_disease_model, animal_disease_features = train_disease_predictor(animal_df, "animal")
    
    crop_model, crop_features = build_model(crop_df, "crop")
    animal_model, animal_features = build_model(animal_df, "animal")
    
    crop_model_file = 'crop_medicine_recommender.pkl'
    animal_model_file = 'animal_medicine_recommender.pkl'
    
    with open(crop_model_file, 'wb') as f:
        pickle.dump({'model': crop_model, 'features': crop_features}, f)
    with open(animal_model_file, 'wb') as f:
        pickle.dump({'model': animal_model, 'features': animal_features}, f)
    
    print("\nModel saved to", crop_model_file)
    print("Model saved to", animal_model_file)
    
    crop_example = {
        "type": "crop",
        "region": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "symptom": "Reddish-brown pustules on leaves",
        "disease": "Leaf Rust"
    }
    
    animal_example = {
        "type": "animal",
        "region": "Punjab",
        "animal": "Cow",
        "age_group": "Adult",
        "symptom": "Swollen udder",
        "disease": "Mastitis"
    }
    
    crop_example_incomplete = {
        "type": "crop",
        "region": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "symptom": "Reddish-brown pustules on leaves"
    }
    
    animal_example_chick = {
        "type": "animal",
        "region": "Punjab",
        "animal": "Poultry",
        "age_group": "Chick",
        "symptom": "Diarrhea",
        "disease": "Coccidiosis"
    }
    
    print("\nCrop Prediction Example:")
    print(predict_medicine(crop_example, crop_model_file, animal_model_file))
    
    print("\nAnimal Prediction Example:")
    print(predict_medicine(animal_example, crop_model_file, animal_model_file))
    
    print("\nCrop Incomplete Prediction Example:")
    print(predict_medicine(crop_example_incomplete, crop_model_file, animal_model_file))
    
    print("\nAnimal Chick Prediction Example:")
    print(predict_medicine(animal_example_chick, crop_model_file, animal_model_file))

if __name__ == "__main__":
    main()