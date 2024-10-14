import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os

# Get the directory of the current script
script_dir = r"C:\Users\SidK\Documents\Documents\HSMA\Hackathon"


# import data set
df = pd.read_csv(r"C:\Users\SidK\Documents\Documents\HSMA\Hackathon\Combined Data.csv") 

# Drop the 'Unnamed: 0' column as it is not needed
df = df.drop(columns=['Unnamed: 0'])

# Handle missing values
df['statement'] = df['statement'].fillna('')

# Extract features and target
X = df['statement']
y = df['status']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert text data to feature vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the Multinomial Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Plot confusion matrix
plt.figure(figsize=(12, 8))  # Increase the figure size
cm_display.plot(cmap=plt.cm.Blues)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=0)  # Optionally, rotate y-axis labels

plt.title("Confusion Matrix")
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()

# Define file paths (saving in the parent directory)
model_path = os.path.join(script_dir,  'model.pkl')
vectorizer_path = os.path.join(script_dir,  'tfidf_vectorizer.pkl')
encoder_path = os.path.join(script_dir, 'label_encoder.pkl')

# Save the model and vectorizer
joblib.dump(model, model_path)
joblib.dump(tfidf_vectorizer, vectorizer_path)
joblib.dump(label_encoder, encoder_path)

