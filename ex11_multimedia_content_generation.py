from transformers import pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import torch

def main():
    # Detect GPU/CPU for Stable Diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    topic = "The benefits of renewable energy"
    print(f"Topic: '{topic}'")

    # 1. Text generation
    print("\n--- 1. Text Generation ---")
    print("Loading generator model 'google/flan-t5-base'...")
    text_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    text_prompt = f"Write a short, engaging paragraph about: {topic}"
    
    print(f"Prompt: '{text_prompt}'")
    generated_text = text_generator(text_prompt, max_length=80, pad_token_id=text_generator.tokenizer.eos_token_id)[0]["generated_text"]
    print("Generated Text:\n", generated_text)

    # 2. Image generation (derived prompt)
    print("\n--- 2. Image Generation ---")
    image_prompt = f"An illustration representing {topic}, digital art"
    print(f"Image Prompt: '{image_prompt}'")
    print(f"Loading Stable Diffusion v1.5 on device: {device}...")
    
    try:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch_dtype
        ).to(device)
        
        print("Generating image...")
        image = sd_pipe(image_prompt, num_inference_steps=25).images[0]
        image.save("content_image.png")
        print("Image saved as content_image.png")
    except Exception as e:
        print(f"Stable Diffusion image generation failed or skipped: {e}")
        print("Creating placeholder image 'content_image.png' instead...")
        from PIL import Image
        placeholder = Image.new("RGB", (300, 300), color="blue")
        placeholder.save("content_image.png")
        print("Placeholder image saved.")

    # 3. Audio generation (text-to-speech)
    print("\n--- 3. Audio Generation ---")
    print("Generating speech using gTTS...")
    tts = gTTS(text=generated_text, lang="en")
    tts.save("content_audio.mp3")
    print("Audio saved as content_audio.mp3")

if __name__ == "__main__":
    main()
