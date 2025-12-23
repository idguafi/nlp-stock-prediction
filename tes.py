import torch
from transformers import pipeline

# 1. Determine the device (MPS for M3 GPU, otherwise CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the sentiment analysis pipeline
# By default, this uses a distilled BERT model (DistilBERT) 
# which is faster and lighter for local use.
classifier = pipeline("text-classification", model="ProsusAI/finbert")

# 3. Test data
data = [
    "I absolutely love the performance of the M3 chip!",
    "Apple Intelligence is just the company's latest effort to recapture its magic from years past.",
    "The Oracle of Omaha has concentrated Berkshire's investment portfolio in his best ideas"
]

# 4. Run inference
results = classifier(data)


# 5. Print results
for text, result in zip(data, results):
    print(f"\nText: {text}")
    print(f"Sentiment: {result} (Score: {result['score']:.4f})")