import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(42)
np.random.seed(42)

# Load dataset
dataset = load_dataset("civil_comments")["train"]
dataset = dataset.shuffle(seed=42).select(range(1000))

texts = dataset["text"]
labels = [1 if t > 0.5 else 0 for t in dataset["toxicity"]]

split = int(0.8 * len(texts))
train_texts, test_texts = texts[:split], texts[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Baseline
toxic_keywords = ["idiot", "stupid", "hate", "dumb", "moron"]

def baseline_predict(text):
    text = text.lower()
    return int(any(k in text for k in toxic_keywords))

baseline_preds = [baseline_predict(t) for t in test_texts]

# AI Pipeline
model_name = "distilbert-base-uncased-finetuned-jigsaw-toxic-comment-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def model_predict(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).tolist()

model_preds = model_predict(test_texts)

# Evaluation
print("Baseline Accuracy:", accuracy_score(test_labels, baseline_preds))
print("Baseline F1:", f1_score(test_labels, baseline_preds))
print("Pipeline Accuracy:", accuracy_score(test_labels, model_preds))
print("Pipeline F1:", f1_score(test_labels, model_preds))

# Qualitative examples
print("\nExamples where predictions differ:\n")
for i in range(len(test_texts)):
    if baseline_preds[i] != model_preds[i]:
        print("Text:", test_texts[i])
        print("True:", test_labels[i])
        print("Baseline:", baseline_preds[i])
        print("Pipeline:", model_preds[i])
        print("-" * 60)
