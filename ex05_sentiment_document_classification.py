from transformers import pipeline

def main():
    # ---------- Sentiment Analysis ----------
    print("Loading sentiment analysis pipeline...")
    # Note: Default model is distilbert-base-uncased-finetuned-sst-2-english
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    reviews = [
        "The new smartphone has an amazing camera and battery life!",
        "The delivery was late and the packaging was damaged."
    ]
    
    print("\n--- Sentiment Analysis Results ---")
    for review in reviews:
        result = sentiment_analyzer(review)[0]
        print(f"Review: '{review}'")
        print(f"Sentiment: {result['label']} (Score: {round(result['score'], 3)})\n")

    # ---------- Document Classification (Zero-Shot) ----------
    print("="*40 + "\n")
    print("Loading zero-shot classification pipeline with facebook/bart-large-mnli...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    document = "The central bank raised interest rates to control rising inflation."
    candidate_labels = ["Politics", "Economy", "Sports", "Technology"]
    
    print(f"\nDocument to classify: '{document}'")
    print(f"Candidate labels: {candidate_labels}")
    
    classification = classifier(document, candidate_labels)
    
    print("\n--- Classification Probabilities ---")
    for label, score in zip(classification["labels"], classification["scores"]):
        print(f"{label}: {round(score, 3)}")

if __name__ == "__main__":
    main()
