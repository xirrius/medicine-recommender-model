import pandas as pd
import numpy as np

def generate_crop_dataset(n_samples):
    crop_medicines = [
        'Mancozeb', 'Carbendazim', 'Propiconazole', 'Azoxystrobin', 'Metalaxyl',
        'Chlorothalonil', 'Tebuconazole', 'Hexaconazole', 'Cymoxanil', 'Pencycuron',
        'Thiram', 'Copper oxychloride', 'Streptocycline', 'Validamycin', 'Trichoderma viride',
        'Sulfur spray', 'Pyraclostrobin', 'Difenoconazole', 'Triadimefon', 'Myclobutanil',
        'Plantomycin', 'Tricyclazole', 'Isoprothiolane', 'Fludioxonil', 'Thiophanate methyl',
        'Vitavax', 'Zineb'
    ]
    diseases = ['Leaf Rust', 'Blast', 'Powdery Mildew', 'Leaf Spot', 'Blight', 'Fusarium Wilt', 'Anthracnose']
    disease_medicine_map = {
        'Leaf Rust': 'Tebuconazole',
        'Blast': 'Isoprothiolane',
        'Powdery Mildew': 'Azoxystrobin',
        'Leaf Spot': 'Mancozeb',
        'Blight': 'Metalaxyl',
        'Fusarium Wilt': 'Carbendazim',
        'Anthracnose': 'Chlorothalonil'
    }
    min_samples_per_class = max(15, n_samples // len(crop_medicines))  # ~18–19 samples per class
    
    # Initialize with balanced classes
    samples_per_class = max(15, n_samples // len(crop_medicines))
    data = []
    for med in crop_medicines:
        for _ in range(samples_per_class):
            disease = next((d for d, m in disease_medicine_map.items() if m == med), np.random.choice(diseases))
            data.append({
                'type': 'crop',
                'region': np.random.choice(['Punjab', 'Tamil Nadu', 'Haryana', 'Gujarat']),
                'crop': np.random.choice(['Wheat', 'Rice', 'Cotton', 'Maize']),
                'season': np.random.choice(['Kharif', 'Rabi']),
                'symptom': np.random.choice(['Yellowing leaves', 'Reddish-brown pustules', 'Stunted growth', 'Spots on leaves']),
                'disease': disease,
                'recommended_medicine': med
            })
    
    df = pd.DataFrame(data)
    
    # Trim or add samples to reach n_samples
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=42)
    elif len(df) < n_samples:
        extra_count = n_samples - len(df)
        extra = pd.DataFrame({
            'type': ['crop'] * extra_count,
            'region': np.random.choice(['Punjab', 'Tamil Nadu', 'Haryana', 'Gujarat'], extra_count),
            'crop': np.random.choice(['Wheat', 'Rice', 'Cotton', 'Maize'], extra_count),
            'season': np.random.choice(['Kharif', 'Rabi'], extra_count),
            'symptom': np.random.choice(['Yellowing leaves', 'Reddish-brown pustules', 'Stunted growth', 'Spots on leaves'], extra_count),
            'disease': np.random.choice(diseases, extra_count),
            'recommended_medicine': np.random.choice(crop_medicines, extra_count)
        })
        df = pd.concat([df, extra], ignore_index=True)
        df = df.sample(n_samples, random_state=42)
    
    # Ensure minimum samples per class
    for med in crop_medicines:
        current_count = len(df[df['recommended_medicine'] == med])
        if current_count < min_samples_per_class:
            extra_count = min_samples_per_class - current_count
            extra = pd.DataFrame({
                'type': ['crop'] * extra_count,
                'region': np.random.choice(['Punjab', 'Tamil Nadu', 'Haryana', 'Gujarat'], extra_count),
                'crop': np.random.choice(['Wheat', 'Rice', 'Cotton', 'Maize'], extra_count),
                'season': np.random.choice(['Kharif', 'Rabi'], extra_count),
                'symptom': np.random.choice(['Yellowing leaves', 'Reddish-brown pustules', 'Stunted growth', 'Spots on leaves'], extra_count),
                'disease': np.random.choice(diseases, extra_count),
                'recommended_medicine': [med] * extra_count
            })
            df = pd.concat([df, extra], ignore_index=True)
            df = df.sample(n_samples, random_state=42)
    
    return df

def generate_animal_dataset(n_samples):
    animal_medicines = [
        'Oxytetracycline', 'Antibiotics', 'Antipyretics', 'Penicillin', 'ND Vaccine',
        'Amprolium', 'Enrofloxacin', 'FMD Vaccine', 'Ceftiofur', 'PPR Vaccine',
        'Amoxicillin-clavulanic acid', 'Toltrazuril', 'Diclazuril', 'Supportive therapy',
        'Poloxalene', 'Simethicone', 'Mineral oil', 'Formaldehyde solution', 'Pirlimycin',
        'Topical antiseptics', 'Vitamins', 'Zinc sulfate', 'Streptomycin', 'Oseltamivir',
        'NSAIDs', 'HS Vaccine', 'BQ Vaccine', 'AI Vaccine', 'Brucella Vaccine',
        'Enterotoxemia Vaccine', 'Ecthyma Vaccine', 'Bluetongue Vaccine', 'Sheep Pox Vaccine',
        'Antitoxin'
    ]
    diseases = ['Mastitis', 'Foot Rot', 'Coccidiosis', 'FMD', 'Pneumonia', 'Bloat', 'Parasitic Infection']
    disease_medicine_map = {
        'Mastitis': 'Pirlimycin',
        'Foot Rot': 'Oxytetracycline',
        'Coccidiosis': 'Amprolium',
        'FMD': 'FMD Vaccine',
        'Pneumonia': 'Antibiotics',
        'Bloat': 'Poloxalene',
        'Parasitic Infection': 'Toltrazuril'
    }
    min_samples_per_class = max(15, n_samples // len(animal_medicines))  # ~14–15 samples per class
    
    # Initialize with balanced classes
    samples_per_class = max(15, n_samples // len(animal_medicines))
    data = []
    for med in animal_medicines:
        for _ in range(samples_per_class):
            disease = next((d for d, m in disease_medicine_map.items() if m == med), np.random.choice(diseases))
            data.append({
                'type': 'animal',
                'region': np.random.choice(['Punjab', 'Tamil Nadu', 'Haryana', 'West Bengal']),
                'animal': np.random.choice(['Cow', 'Buffalo', 'Goat', 'Poultry']),
                'age_group': np.random.choice(['Adult', 'Young', 'Senior', 'Chick']),
                'symptom': np.random.choice(['Swollen udder', 'Fever', 'Diarrhea', 'Lameness']),
                'disease': disease,
                'recommended_medicine': med
            })
    
    df = pd.DataFrame(data)
    
    # Trim or add samples to reach n_samples
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=42)
    elif len(df) < n_samples:
        extra_count = n_samples - len(df)
        extra = pd.DataFrame({
            'type': ['animal'] * extra_count,
            'region': np.random.choice(['Punjab', 'Tamil Nadu', 'Haryana', 'West Bengal'], extra_count),
            'animal': np.random.choice(['Cow', 'Buffalo', 'Goat', 'Poultry'], extra_count),
            'age_group': np.random.choice(['Adult', 'Young', 'Senior', 'Chick'], extra_count),
            'symptom': np.random.choice(['Swollen udder', 'Fever', 'Diarrhea', 'Lameness'], extra_count),
            'disease': np.random.choice(diseases, extra_count),
            'recommended_medicine': np.random.choice(animal_medicines, extra_count)
        })
        df = pd.concat([df, extra], ignore_index=True)
        df = df.sample(n_samples, random_state=42)
    
    # Ensure minimum samples per class
    for med in animal_medicines:
        current_count = len(df[df['recommended_medicine'] == med])
        if current_count < min_samples_per_class:
            extra_count = min_samples_per_class - current_count
            extra = pd.DataFrame({
                'type': ['animal'] * extra_count,
                'region': np.random.choice(['Punjab', 'Tamil Nadu', 'Haryana', 'West Bengal'], extra_count),
                'animal': np.random.choice(['Cow', 'Buffalo', 'Goat', 'Poultry'], extra_count),
                'age_group': np.random.choice(['Adult', 'Young', 'Senior', 'Chick'], extra_count),
                'symptom': np.random.choice(['Swollen udder', 'Fever', 'Diarrhea', 'Lameness'], extra_count),
                'disease': np.random.choice(diseases, extra_count),
                'recommended_medicine': [med] * extra_count
            })
            df = pd.concat([df, extra], ignore_index=True)
            df = df.sample(n_samples, random_state=42)
    
    return df

if __name__ == "__main__":
    crop_df = generate_crop_dataset(500)
    animal_df = generate_animal_dataset(500)
    crop_df.to_csv('crop_medicine_dataset.csv', index=False)
    animal_df.to_csv('animal_medicine_dataset.csv', index=False)
    print("Generated and saved datasets: crop_medicine_dataset.csv, animal_medicine_dataset.csv")