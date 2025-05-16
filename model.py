import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset (replace with your actual CSV path)
df = pd.read_csv('train_balanced.csv')

# Step 1: Text Embedding Generation (using MiniLM)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate sentence embeddings
def embed_texts(texts, batch_size=500):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch)
        embeddings.extend(emb)
    return embeddings

# Example usage (embedding discourse_text column)
discourse_texts = df['discourse_text'].tolist()  # Get your discourse text
embeddings = embed_texts(discourse_texts)

# Step 2: Effectiveness Classification (using SVM)
# Encode the target variable (discourse_effectiveness)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['discourse_effectiveness'])

# Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.25, random_state=42)

# Initialize and train the SVM classifier
clf = SVC(kernel='linear', class_weight='balanced')
clf.fit(X_train, y_train)

# Step 3: Evaluate the model
y_pred = clf.predict(X_test)

# Print classification report
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Calculate accuracy, F1 score, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"[INFO] Accuracy: {accuracy:.4f}")
print(f"[INFO] F1 Score: {f1:.4f}")
print(f"[INFO] Precision: {precision:.4f}")
print(f"[INFO] Recall: {recall:.4f}")

# Save the trained model and label encoder
joblib.dump(clf, 'effectiveness_classifier_svm.pk2')
joblib.dump(label_encoder, 'label_encoder.pk2')
