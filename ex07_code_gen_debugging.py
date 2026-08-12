from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Global references for model and tokenizer
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading Salesforce/codegen-350M-mono tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

def generate_code(prompt, max_new_tokens=80):
    load_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate code using codegen-350M-mono
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # 1. Code generation from a natural-language instruction
    prompt1 = "# Write a Python function to check if a number is prime\ndef is_prime(n):"
    print(f"\n--- Running Task 1: Code Generation ---")
    print(f"Prompt:\n{prompt1}\n")
    print("Generating function...")
    generated = generate_code(prompt1)
    print("Generated Function:\n", generated)

    # 2. Debugging a faulty snippet
    buggy_code = """# The following function should return the factorial of n, but has a bug. Fix it.
def factorial(n):
    result = 0
    for i in range(1, n+1):
        result = result * i
    return result
# Corrected function:
def factorial_fixed(n):"""
    
    print(f"\n" + "="*40 + "\n")
    print(f"--- Running Task 2: Code Debugging ---")
    print(f"Buggy Code Prompt:\n{buggy_code}\n")
    print("Generating fix...")
    fixed = generate_code(buggy_code, max_new_tokens=60)
    print("Debug Suggestion:\n", fixed)

if __name__ == "__main__":
    main()
