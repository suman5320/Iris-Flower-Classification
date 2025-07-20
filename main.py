from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from joblib import load
import numpy as np

app = FastAPI()

# Load the trained model
model = load("svm_model.joblib")

# Target labels
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Define input schema using Pydantic
class FlowerFeatures(BaseModel):
    features: list[float] = [5.1, 1.4, 0.2]
# Only 3 features since 'sepal width' was dropped
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/predict")
def predict_species(data: FlowerFeatures):
    try:
        # Convert to NumPy array
        features_array = np.array(data.features).reshape(1, -1)

        # Predict
        prediction = model.predict(features_array)[0]
        class_name = target_names[prediction]

        return {
            "prediction": int(prediction),
            "class_name": class_name
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
