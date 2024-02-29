import torch
from generator_model import Generator
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import config

def load_generator(checkpoint_path, device='cuda'):
    """Load the trained generator model."""
    model = Generator(img_channels=3, num_residuals=9).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def transform_image(image_path):
    """Transform the input image."""
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def generate_image(model, input_image, device='cuda'):
    """Generate an image using the model."""
    with torch.no_grad():
        input_tensor = transform_image(input_image).to(device)
        generated_tensor = model(input_tensor)
        save_image(generated_tensor * 0.5 + 0.5, "generated_image.png")

if __name__ == "__main__":
    device = config.DEVICE
    # Load the generator model. Choose the correct checkpoint for the desired direction.
    gen_Z_checkpoint = config.CHECKPOINT_GEN_Z  # For generating zebra images from horse images
    # Or use gen_H_checkpoint for horse images from zebra images.
    generator = load_generator(gen_Z_checkpoint, device)
    
    # Path to your input image
    input_image_path = r"val\n02381460_20.jpg"
    
    generate_image(generator, input_image_path, device)
