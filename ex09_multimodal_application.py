from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image
import requests

def main():
    # Use a public image URL (dog in a field or similar)
    image_url = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"
    print(f"Fetching image from URL: {image_url}...")
    try:
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    except Exception as e:
        print(f"Failed to fetch image from URL: {e}")
        print("Creating a placeholder image for testing...")
        raw_image = Image.new("RGB", (300, 300), color="green")

    # ---------- Image Captioning ----------
    print("\n--- Image Captioning Task ---")
    print("Loading Salesforce/blip-image-captioning-base model...")
    cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    print("Processing image for captioning...")
    inputs = cap_processor(raw_image, return_tensors="pt")
    caption_ids = cap_model.generate(**inputs, max_new_tokens=30)
    caption = cap_processor.decode(caption_ids[0], skip_special_tokens=True)
    print("Generated Caption:", caption)

    # ---------- Visual Question Answering ----------
    print("\n" + "="*40 + "\n")
    print("--- Visual Question Answering Task ---")
    print("Loading Salesforce/blip-vqa-base model...")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    question = "What animal is in the picture?"
    print(f"Question: {question}")
    
    print("Processing image and question...")
    vqa_inputs = vqa_processor(raw_image, question, return_tensors="pt")
    answer_ids = vqa_model.generate(**vqa_inputs)
    answer = vqa_processor.decode(answer_ids[0], skip_special_tokens=True)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
