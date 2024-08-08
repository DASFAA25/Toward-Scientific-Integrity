import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('Classification.csv', sep=';', encoding='latin1')

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['citation'], df['label'], test_size=0.2,
                                                                      random_state=42)

# Use TF-IDF vectorizer for text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Scale the data for other classifiers
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to dense format for classifiers that require dense input
X_train_dense = X_train_scaled.toarray()
X_test_dense = X_test_scaled.toarray()

# Define classifiers (with dense-compatible ones using dense matrices)
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'SVM': SVC(kernel='linear'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': MultinomialNB(),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'LDA': LDA(),
    'QDA': QDA()
}

results = {}
wrongly_classified = {}

# Evaluate and collect results for classifiers
for name, clf in classifiers.items():
    if name in ['LDA', 'QDA', 'Naive Bayes']:
        # Use dense matrices for these classifiers
        clf.fit(X_train_dense, train_labels)
        y_pred = clf.predict(X_test_dense)
    else:
        clf.fit(X_train_scaled, train_labels)
        y_pred = clf.predict(X_test_scaled)

    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='binary', pos_label=1)
    accuracy = accuracy_score(test_labels, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'confusion_matrix': confusion_matrix(test_labels, y_pred)
    }

    misclassified_indices = [index for index, (true, pred) in enumerate(zip(test_labels, y_pred)) if true != pred]
    wrongly_classified_examples = {
        'sentence': test_texts.iloc[misclassified_indices].values,
        'actual_label': np.array(test_labels)[misclassified_indices],
        'predicted_label': np.array(y_pred)[misclassified_indices]
    }
    wrongly_classified[name] = wrongly_classified_examples

# Print evaluation metrics and confusion matrix
for name, metrics in results.items():
    print(f"Classifier: {name}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1-score: {metrics['f1-score']}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\n")

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'CC'],
                yticklabels=['NC', 'CC'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Print wrongly classified citation sentences
for name, examples in wrongly_classified.items():
    print(f"Wrongly Classified Citations for {name}:")
    for sentence, actual, predicted in zip(examples['sentence'], examples['actual_label'], examples['predicted_label']):
        print(f"Sentence: {sentence}")
        print(f"Actual Label: {label_encoder.inverse_transform([actual])[0]}")
        print(f"Predicted Label: {label_encoder.inverse_transform([predicted])[0]}")
        print("\n")
