import pandas as pd
import json

df = pd.read_csv("ml/Training.csv")

disease_symptom_map = {}

for _, row in df.iterrows():
    disease = row[0].strip()
    symptoms = set([str(s).strip().lower().replace(" ", "_") for s in row[1:] if pd.notna(s)])
    
    if disease not in disease_symptom_map:
        disease_symptom_map[disease] = []
    
    disease_symptom_map[disease].append(symptoms)

# Save to JSON
with open("ml/rules.json", "w") as f:
    json.dump({k: [list(s) for s in v] for k, v in disease_symptom_map.items()}, f)
