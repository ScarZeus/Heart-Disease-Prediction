from sklearn.linear_model import LogisticRegression
import joblib

def build_model():
    model = LogisticRegression()
    return model

def save_model(model, scaler, path):
    joblib.dump({'model': model, 'scaler': scaler}, path)

def load_model(path):
    data = joblib.load(path)
    return data['model'], data['scaler']
