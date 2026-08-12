from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

def main():
    # 1. Knowledge base
    documents = [
        "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "Retrieval-Augmented Generation combines document retrieval with text generation.",
        "Python is a popular high-level programming language used in AI development.",
        "Vector databases store embeddings and support fast similarity search."
    ]

    print("--- Knowledge Base Documents ---")
    for idx, doc in enumerate(documents, 1):
        print(f"{idx}. {doc}")

    # 2. Embed documents
    print("\nLoading sentence transformer model 'all-MiniLM-L6-v2' and generating embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embed_model.encode(documents)

    # 3. Build FAISS index
    print("Building FAISS index (IndexFlatL2)...")
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    # 4. Query and retrieve top-2 relevant chunks
    query = "What is RAG in AI?"
    print(f"\nQuerying: '{query}'")
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=2)
    retrieved_chunks = [documents[i] for i in I[0]]

    print("\nRetrieved Context Chunks (Top-2 closest):")
    for chunk in retrieved_chunks:
        print(f"- {chunk}")

    # 5. Build augmented prompt and generate answer
    context = " ".join(retrieved_chunks)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

    print("\nLoading generator model 'google/flan-t5-base' and generating grounded response...")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    # To avoid warning, specify pad_token_id
    answer = generator(prompt, max_length=60, pad_token_id=generator.tokenizer.eos_token_id)
    
    print("\n=== RAG Answer ===")
    print(answer[0]["generated_text"])

if __name__ == "__main__":
    main()
