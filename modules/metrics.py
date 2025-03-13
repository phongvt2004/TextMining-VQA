from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def compute_metrics(logits, labels):
    predictions = np.argmax(logits, axis=-1)  # Get class with highest probability

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)

    # Exact Match (EM) Score
    exact_match = np.mean([1 if p == l else 0 for p, l in zip(predictions, labels)])

    # BLEU Score (Unigram BLEU-1)
    smooth_fn = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([str(l)], str(p), smoothing_function=smooth_fn) for p, l in zip(predictions, labels)]
    bleu_score = np.mean(bleu_scores)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "bleu_score": bleu_score
    }