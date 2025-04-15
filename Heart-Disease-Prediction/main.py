from src.predict import predict_heart_disease

def get_user_input():
    print("Enter the following details for heart disease prediction:")
    user_input = {}
    user_input['age'] = float(input("Age (years): "))
    user_input['sex'] = int(input("Sex (1 = male, 0 = female): "))
    user_input['cp'] = int(input("Chest Pain Type (0-3): "))
    user_input['trestbps'] = float(input("Resting Blood Pressure (mm Hg): "))
    user_input['chol'] = float(input("Cholesterol (mg/dl): "))
    user_input['fbs'] = int(input("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false): "))
    user_input['restecg'] = int(input("Resting ECG result (0-2): "))
    user_input['thalach'] = float(input("Maximum Heart Rate achieved: "))
    user_input['exang'] = int(input("Exercise Induced Angina (1 = yes, 0 = no): "))
    user_input['oldpeak'] = float(input("ST Depression induced by exercise: "))
    user_input['slope'] = int(input("Slope of the peak exercise ST segment (0-2): "))
    user_input['ca'] = int(input("Number of major vessels (0-3): "))
    user_input['thal'] = int(input("Thal (0 = normal, 1 = fixed defect, 2 = reversible defect): "))
    
    return user_input

if __name__ == "__main__":
    user_input = get_user_input()
    model_path = "models/heart_model.pkl"
    prediction = predict_heart_disease(user_input, model_path)
    if prediction == 1:
        print("\nHeart Disease Detected! Please consult a doctor for further examination.")
    else:
        print("\nNo Heart Disease Detected. Keep maintaining a healthy lifestyle!")
