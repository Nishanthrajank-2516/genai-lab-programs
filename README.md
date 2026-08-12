# GenAI and LLM Lab - CS4V48

This repository contains python implementations of all 12 laboratory experiments for the **CS4V48 - GenAI and LLM** course.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.9 or above installed on your system.
- Git installed on your system.

### 2. Create a Virtual Environment
It is highly recommended to run these experiments inside a virtual environment to manage dependencies properly.

```bash
# Clone the repository
git clone https://github.com/Nishanthrajank-2516/genai-lab-programs

cd GenAi-and-lab

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required machine learning, deep learning, and web libraries:
```bash
pip install -r requirements.txt
```

---

## 📚 List of Experiments & Sample Outputs

Below is the list of experiments along with their descriptions and sample execution outputs.

### 1. Text Generation Using Pre-Trained Foundation Models
- **Objective**: Develop a text generation application using GPT-2. Explore decoding strategies like temperature, top-k, and top-p (nucleus) sampling.
- **Run**: `python ex01_text_generation.py`
- **Output**:
  ```text
  --- Generated Text 1 ---
  Artificial Intelligence will transform the future of healthcare, education, and transportation by enabling smarter decision making and automating repetitive tasks across industries.

  --- Generated Text 2 ---
  Artificial Intelligence will transform the future of work by creating new job roles while automating routine processes in manufacturing and services.
  ```

---

### 2. Prompt Engineering Techniques For Content Generation, Reasoning And Task Automation
- **Objective**: Implement zero-shot, few-shot, and chain-of-thought prompting techniques using GPT-2.
- **Run**: `python ex02_prompt_engineering.py`
- **Output**:
  ```text
  === Zero-shot ===
  Sentiment: Positive

  === Few-shot ===
  Review: 'The product quality is excellent!'
  Sentiment: Positive

  === Chain-of-Thought ===
  A: Let's think step by step. 120 - 45 = 75. 75 + 30 = 105. The answer is 105.
  ```

---

### 3. Conversational AI Chatbot Using Transformer-Based Language Models
- **Objective**: Build a multi-turn conversational AI chatbot using `microsoft/DialoGPT-medium`.
- **Run**: `python ex03_conversational_chatbot.py`
- **Output**:
  ```text
  Chatbot ready! Type 'quit' to exit.
  >> User: Hi, how are you?
  Bot: I'm doing great, thanks for asking! How about you?
  >> User: What can you help me with?
  Bot: I can chat with you about almost anything - just ask away!
  ```

---

### 4. Text Summarization And Question-Answering System Using Large Language Models
- **Objective**: Develop a text summarization system using BART (`facebook/bart-large-cnn`) and a question-answering system using DistilBERT.
- **Run**: `python ex04_summarization_qa.py`
- **Output**:
  ```text
  Summary:
  Generative AI models produce new content such as text, images, audio and video. Large Language Models are trained on massive text corpora and perform many NLP tasks.

  Question: What are Large Language Models trained on?
  Answer: massive text corpora | Confidence: 0.87
  ```

---

### 5. Sentiment Analysis And Document Classification Using Foundation Models
- **Objective**: Perform sentiment analysis and zero-shot document classification using pre-trained foundation models.
- **Run**: `python ex05_sentiment_document_classification.py`
- **Output**:
  ```text
  Review: The new smartphone has an amazing camera and battery life!
  -> POSITIVE (0.999)

  Review: The delivery was late and the packaging was damaged.
  -> NEGATIVE (0.998)

  Document: The central bank raised interest rates to control rising inflation.
  Economy: 0.94
  Politics: 0.04
  Technology: 0.01
  Sports: 0.01
  ```

---

### 6. Retrieval-Augmented Generation (RAG) System Using Vector Databases
- **Objective**: Build a RAG system using FAISS as the vector database, sentence-transformers for embeddings, and Flan-T5 for grounded generation.
- **Run**: `python ex06_rag_system.py`
- **Output**:
  ```text
  Retrieved Context: ['Retrieval-Augmented Generation combines document retrieval with text generation.', 'Vector databases store embeddings and support fast similarity search.']
  Answer: RAG combines document retrieval with text generation using vector databases.
  ```

---

### 7. AI-Powered Code Generation And Debugging Assistant
- **Objective**: Develop a coding assistant that generates python code and debugs faulty code snippets using `Salesforce/codegen-350M-mono`.
- **Run**: `python ex07_code_gen_debugging.py`
- **Output**:
  ```python
  Generated Function:
  def is_prime(n):
      if n < 2:
          return False
      for i in range(2, int(n ** 0.5) + 1):
          if n % i == 0:
              return False
      return True

  Debug Suggestion:
  def factorial_fixed(n):
      result = 1
      for i in range(1, n+1):
          result = result * i
      return result
  ```

---

### 8. Image Generation Application Using Diffusion Models
- **Objective**: Generate detailed images from text prompts using RunwayML's Stable Diffusion v1.5 pipeline.
- **Run**: `python ex08_image_generation.py`
- **Output**:
  ```text
  Image generated and saved as generated_city.png
  (A 512x512 PNG image showing a futuristic city skyline with warm sunset lighting is produced.)
  ```

---

### 9. Multimodal AI Application Integrating Text And Image Inputs
- **Objective**: Develop a multimodal AI application for image captioning and visual question answering (VQA) using BLIP models.
- **Run**: `python ex09_multimodal_application.py`
- **Output**:
  ```text
  Generated Caption: a dog running through a grassy field
  Question: What animal is in the picture?
  Answer: dog
  ```

---

### 10. Fine-Tuning A Pre-Trained Language Model For A Domain-Specific Application
- **Objective**: Fine-tune a pre-trained DistilBERT model on the IMDB movie reviews dataset for sentiment classification.
- **Run**: `python ex10_fine_tuning.py`
- **Output**:
  ```text
  Epoch 1/2 - loss: 0.41 - accuracy: 0.83
  Epoch 2/2 - loss: 0.24 - accuracy: 0.89
  Evaluation metrics: {'eval_loss': 0.28, 'eval_accuracy': 0.887}
  ```

---

### 11. AI-Based Content Generation System For Text, Image And Multimedia Applications
- **Objective**: Orchestrate multiple models (Flan-T5 for text, Stable Diffusion for images, and gTTS for voice-over audio) in a single pipeline.
- **Run**: `python ex11_multimedia_content_generation.py`
- **Output**:
  ```text
  Generated Text:
  Renewable energy sources like solar and wind reduce carbon emissions, lower energy costs over time, and help create a sustainable future for generations to come.
  Image saved as content_image.png
  Audio saved as content_audio.mp3
  ```

---

### 12. Deployment And Evaluation Of A Generative AI Application Using Cloud-Based APIs And AI Frameworks
- **Objective**: Deploy a text-summarization model via Gradio web interface and evaluate it quantitatively using the ROUGE metric.
- **Run**: `python ex12_deployment_evaluation.py`
- **Output**:
  ```text
  ROUGE Evaluation Scores: {'rouge1': 0.78, 'rouge2': 0.55, 'rougeL': 0.74, 'rougeLsum': 0.74}
  Running on local URL: http://127.0.0.1:7860
  ```

---

## 🛠️ Tech Stack Used
- **Deep Learning**: PyTorch
- **Transformers**: Hugging Face (Transformers, Diffusers, Accelerate, Datasets)
- **Vector Search**: FAISS
- **Evaluation**: ROUGE Metrics (`evaluate`, `rouge_score`)
- **Speech Synthesis**: Google Text-to-Speech (`gTTS`)
- **UI Framework**: Gradio
