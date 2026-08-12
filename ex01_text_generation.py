from transformers import pipeline, set_seed

def main():
    print("Loading pre-trained GPT-2 text generation pipeline...")
    # Load the pre-trained GPT-2 text generation pipeline
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)

    prompt = "Artificial Intelligence will transform the future of"
    print(f"\nPrompt: {prompt}\n")

    # Generate text
    print("Generating text using sampling-based decoding (temp=0.8, top_k=50, top_p=0.95)...")
    outputs = generator(
        prompt,
        max_length=60,
        num_return_sequences=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    print("\n--- Outputs ---")
    for i, out in enumerate(outputs, 1):
        print(f"--- Generated Text {i} ---")
        print(out["generated_text"])
        print()

if __name__ == "__main__":
    main()
