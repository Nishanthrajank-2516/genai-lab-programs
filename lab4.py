from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForQuestionAnswering

#---------- Text Summarization---------
# The 'pipeline' function's 'summarization' or 'text2text-generation' tasks are not recognized
# in this environment. We will manually load the model and tokenizer for summarization.

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

article = """Generative AI refers to a class of artificial intelligence models capable of
producing new content such as text, images, audio, and video. Large Language Models (LLMs)
such as GPT and LLaMA are trained on massive text corpora and can perform a wide range of
natural language tasks including translation, summarization, and question answering. These
models are increasingly being deployed in industry applications ranging from customer support
to software development, transforming how humans interact with machines."""

# Encode the article
inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")

# Generate summary
# Parameters like max_length, min_length, do_sample, etc., are passed to model.generate
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4, # Often used for summarization, a form of beam search
    min_length=20,
    max_length=45,
    do_sample=True
)

summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:\n", summary_text)

#---------- Question Answering---------
# The 'pipeline' function's 'question-answering' task is not recognized in this environment.
# We will manually load the model and tokenizer for question answering.

qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

context = article
question = "name some of the LLM mnodels?"

inputs_qa = qa_tokenizer(question, context, return_tensors="pt")

# Get the raw model output (logits for start and end positions)
outputs_qa = qa_model(**inputs_qa)
start_positions = outputs_qa.start_logits
end_positions = outputs_qa.end_logits

# Find the tokens with the highest `start` and `end` scores
start_index = start_positions.argmax()
end_index = end_positions.argmax() + 1 # +1 to include the end token

# Decode the answer span
answer_tokens = inputs_qa.input_ids[0][start_index:end_index]
answer_text = qa_tokenizer.decode(answer_tokens)

print("\nQuestion:", question)
print("Answer:", answer_text)