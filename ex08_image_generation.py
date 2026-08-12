from diffusers import StableDiffusionPipeline
import torch

def main():
    # Detect GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading RunwayML Stable Diffusion v1.5 pipeline on device: {device}...")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)

    # Disable NSFW filter for faster loading/reduced memory footprints if needed,
    # or keep it default. Let's keep it default.

    prompt = "A futuristic city skyline at sunset, digital art, highly detailed"
    print(f"\nPrompt: '{prompt}'")
    print("Generating image...")
    
    # Generate image
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    filename = "generated_city.png"
    image.save(filename)
    print(f"Image generated and saved successfully as '{filename}'.")

if __name__ == "__main__":
    main()
