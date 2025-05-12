import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

def embed_texts(texts, model, batch_size=500):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="[INFO] Embedding Batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch)
        embeddings.extend(emb)
    return embeddings

def evaluate_model(test_csv):
    print("[INFO] Loading test data...")
    df = pd.read_csv(test_csv)
    df.dropna(subset=['discourse_text', 'discourse_effectiveness'], inplace=True)

    print("[INFO] Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("[INFO] Embedding test texts...")
    embeddings = embed_texts(df['discourse_text'].tolist(), model)

    # Load the trained Ridge model and LabelEncoder
    print("[INFO] Loading saved models...")
    reg = joblib.load('ridge_model.pkl')
    le = joblib.load('label_encoder.pkl')

    print("[INFO] Making predictions...")
    predictions = reg.predict(embeddings)

    # Convert predictions back to original labels using the label encoder
    predicted_labels = le.inverse_transform(predictions.round().astype(int))

    # Add predictions to the dataframe
    df['predicted_effectiveness'] = predicted_labels

    # Evaluate the model performance
    true_labels = df['discourse_effectiveness']
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"[INFO] Accuracy: {accuracy * 100:.2f}%")
    
    # F1 Score (for classification)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"[INFO] F1 Score: {f1:.4f}")
    
    # Classification Report (Precision, Recall, F1 Score per class)
    print("[INFO] Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Optionally, save the predictions to a new CSV file
    df.to_csv('test_predictions_with_metrics.csv', index=False)

if __name__ == "__main__":
    evaluate_model("test.csv")
