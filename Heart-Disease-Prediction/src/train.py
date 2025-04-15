import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import preprocess_data
from model import build_model, save_model

def train_model(data_filepath, model_save_path):
    
    df = load_data(data_filepath)
    
    X, y, scaler = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    
    model = build_model()
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("Training Accuracy: {:.2f}%".format(train_score * 100))
    print("Testing Accuracy: {:.2f}%".format(test_score * 100))
    
    acc = [train_score * 100, test_score * 100]
    labels = ['Training Accuracy', 'Testing Accuracy']
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, acc, color=['blue', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 110)
    plt.title('Model Accuracy')
    
    for bar, acc_value in zip(bars, acc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 2,f"{acc_value:.2f}%", ha='center', fontweight='bold')
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    graph_path = os.path.join(os.path.dirname(model_save_path), "accuracy_graph.png")
    plt.savefig(graph_path)
    plt.close()
    print("Accuracy graph saved to:", graph_path)
    
    save_model(model, scaler, model_save_path)
    print("Model saved to:", model_save_path)

if __name__ == "__main__":
    data_filepath = "data/heart.csv"
    model_save_path = "models/heart_model.pkl"
    train_model(data_filepath, model_save_path)
