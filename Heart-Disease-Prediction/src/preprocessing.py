import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    
    X.fillna(X.mean(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def preprocess_input(user_input, scaler):
    df_input = pd.DataFrame([user_input])
    scaled_input = scaler.transform(df_input)
    return scaled_input
