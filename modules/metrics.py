from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(logits, labels):
    predictions = np.argmax(logits, axis=-1)  # Get class with highest probability

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }