from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

def main():
    print("Loading DialoGPT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    chat_history_ids = None

    print("\nChatbot ready! Type 'quit' to exit.")
    
    # Check if we are running in non-interactive environment (mock input)
    mock_inputs = ["Hi, how are you?", "What can you help me with?", "Tell me a joke", "quit"]
    is_interactive = sys.stdin.isatty()
    
    step = 0
    while step < 5:
        if is_interactive:
            try:
                user_input = input(">> User: ")
            except EOFError:
                break
        else:
            if step < len(mock_inputs):
                user_input = mock_inputs[step]
                print(f">> User: {user_input}")
            else:
                break

        if user_input.lower() == "quit":
            break

        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        
        # Concatenate chat history if exists
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
        
        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )
        
        # Decode and print response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {response}")
        step += 1

if __name__ == "__main__":
    main()
