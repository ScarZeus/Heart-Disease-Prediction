from src.model import load_model
from src.preprocessing import preprocess_input

def predict_heart_disease(user_input, model_path):
    model, scaler = load_model(model_path)
    scaled_input = preprocess_input(user_input, scaler)
    prediction = model.predict(scaled_input)
    return prediction[0]
