from transformers import pipeline

def main():
    # ---------- Text Summarization ----------
    print("Loading summarization pipeline with facebook/bart-large-cnn...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    article = """Generative AI refers to a class of artificial intelligence models capable of
producing new content such as text, images, audio, and video. Large Language Models (LLMs)
such as GPT and LLaMA are trained on massive text corpora and can perform a wide range of
natural language tasks including translation, summarization, and question answering. These
models are increasingly being deployed in industry applications ranging from customer support
to software development, transforming how humans interact with machines."""

    print("\nOriginal Article:")
    print(article)

    print("\nGenerating summary...")
    summary = summarizer(article, max_length=45, min_length=20, do_sample=False)
    print("\nSummary:")
    print(summary[0]["summary_text"])

    # ---------- Question Answering ----------
    print("\n" + "="*40 + "\n")
    print("Loading question-answering pipeline with distilbert-base-cased-distilled-squad...")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    context = article
    question = "What are Large Language Models trained on?"
    
    print(f"\nQuestion: {question}")
    answer = qa(question=question, context=context)
    print("Answer:", answer["answer"])
    print("Confidence Score:", round(answer["score"], 3))

if __name__ == "__main__":
    main()
