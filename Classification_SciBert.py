import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Load the dataset
df = pd.read_csv('Classification.csv', sep=';', encoding='latin1')

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['citation'], df['label'], test_size=0.2,
                                                                      random_state=42)

# Function to preprocess text data using SciBERT tokenizer
def tokenize_function(texts):
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors='pt')


# Tokenize texts
train_encodings = tokenize_function(train_texts.tolist())
test_encodings = tokenize_function(test_texts.tolist())


# Create PyTorch datasets
class CitationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CitationDataset(train_encodings, train_labels.tolist())
test_dataset = CitationDataset(test_encodings, test_labels.tolist())

# Initialize the SciBERT model for sequence classification
scibert_model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2)

# Define training arguments with matched save and eval strategy
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # set both evaluation and save strategy to epoch
    save_strategy="epoch",
    load_best_model_at_end=True  # Load the best model at the end of training
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=scibert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train and evaluate the SciBERT model
trainer.train()
pred_output = trainer.predict(test_dataset)

# Convert predictions to labels
y_test = test_labels.values
y_pred = np.argmax(pred_output.predictions, axis=1)

# Evaluate SciBERT results
results = {}
misclassified_indices = [index for index, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
wrongly_classified_examples = {
    'sentence': test_texts.iloc[misclassified_indices].values,
    'actual_label': y_test[misclassified_indices],
    'predicted_label': y_pred[misclassified_indices]
}

results['SciBERT'] = {
    'precision': precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[0],
    'recall': precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[1],
    'f1-score': precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[2],
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}

# Print evaluation metrics and confusion matrix
for name, metrics in results.items():
    print(f"Classifier: {name}")
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
print(f"Wrongly Classified Citations for SciBERT:")
for sentence, actual, predicted in zip(wrongly_classified_examples['sentence'], wrongly_classified_examples['actual_label'], wrongly_classified_examples['predicted_label']):
    print(f"Sentence: {sentence}")
    print(f"Actual Label: {label_encoder.inverse_transform([actual])[0]}")
    print(f"Predicted Label: {label_encoder.inverse_transform([predicted])[0]}")
    print("\n")
