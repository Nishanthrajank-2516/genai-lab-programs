import gradio as gr
from transformers import pipeline
import evaluate
import time

def main():
    # ---------- 2. Evaluate Generated Output ----------
    print("--- Running NLP Evaluation (ROUGE) ---")
    try:
        rouge = evaluate.load("rouge")
        generated_summaries = [
            "AI models generate new content such as text and images.",
        ]
        reference_summaries = [
            "Generative AI models are capable of producing new content including text and images.",
        ]
        print(f"Generated Summary: '{generated_summaries[0]}'")
        print(f"Reference Summary: '{reference_summaries[0]}'")
        
        scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)
        print("ROUGE Evaluation Scores:", scores)
    except Exception as e:
        print(f"Evaluation failed (requires Internet/evaluate/rouge_score pkg): {e}")

    # ---------- 1. Build and Deploy the App ----------
    print("\n" + "="*40 + "\n")
    print("--- Initializing Gradio Web App ---")
    print("Loading summarizer pipeline with facebook/bart-large-cnn...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_text(input_text):
        if not input_text.strip():
            return "Please enter text to summarize."
        result = summarizer(input_text, max_length=45, min_length=15, do_sample=False)
        return result[0]["summary_text"]

    demo = gr.Interface(
        fn=summarize_text,
        inputs=gr.Textbox(lines=8, label="Enter text to summarize"),
        outputs=gr.Textbox(label="Generated Summary"),
        title="GenAI Text Summarizer",
        description="A cloud-deployable Generative AI summarization app built with Gradio."
    )
    
    print("Launching Gradio web app (share=False for local run)...")
    # Using share=False by default to avoid blocking/issues on restricted environments, 
    # but the user can easily change it to True.
    demo.launch(share=False)

if __name__ == "__main__":
    main()
