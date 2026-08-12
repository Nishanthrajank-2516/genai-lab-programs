from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                        TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import accuracy_score
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device}")

    # 1. Load a domain-specific dataset (example: IMDB movie reviews)
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # If CPU, use a very small subset to run quickly; if GPU, use larger subset
    if device == "cuda":
        train_size, test_size = 2000, 500
        epochs = 2
    else:
        print("GPU not detected. Using a very small subset of dataset for fast CPU run...")
        train_size, test_size = 20, 10
        epochs = 1

    print(f"Selecting {train_size} training samples and {test_size} test samples...")
    small_train = dataset["train"].shuffle(seed=42).select(range(train_size))
    small_test = dataset["test"].shuffle(seed=42).select(range(test_size))

    # 2. Tokenize
    print("Loading distilbert-base-uncased tokenizer and tokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        
    train_ds = small_train.map(tokenize, batched=True)
    test_ds = small_test.map(tokenize, batched=True)

    # 3. Load pre-trained model with classification head
    print("Loading distilbert-base-uncased model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # 4. Training arguments
    print(f"Setting training arguments (epochs={epochs})...")
    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        logging_steps=10 if device == "cpu" else 50,
        no_cuda=(device == "cpu")  # Force cpu if cuda is false
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(pred.label_ids, preds)}

    # 5. Train
    print("Initializing Trainer and starting training...")
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds,
        eval_dataset=test_ds, 
        compute_metrics=compute_metrics
    )
    trainer.train()

    # 6. Evaluate and save
    print("Evaluating fine-tuned model...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
    
    save_path = "./fine_tuned_distilbert_imdb"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
