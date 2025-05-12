import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

def embed_texts(texts, model, batch_size=500):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="[INFO] Embedding Batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch)
        embeddings.extend(emb)
    return embeddings

def train_model(train_csv):
    print("[INFO] Loading training data...")
    df = pd.read_csv(train_csv)
    df.dropna(subset=['discourse_text', 'discourse_effectiveness'], inplace=True)

    print("[INFO] Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("[INFO] Embedding training texts...")
    embeddings = embed_texts(df['discourse_text'].tolist(), model)

    print("[INFO] Encoding target labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['discourse_effectiveness'])

    print("[INFO] Training Ridge Regression model...")
    reg = Ridge(alpha=1.0)
    reg.fit(embeddings, y)

    # Save model and encoder
    joblib.dump(reg, 'ridge_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print("[INFO] Training completed. Model and label encoder saved.")

if __name__ == "__main__":
    train_model("train.csv")
