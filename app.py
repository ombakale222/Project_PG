import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# Device configuration - Define this at the top of your script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the VGG class as before
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# Function to load and transform an image
def load_image(image):
    imsize = 356
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])
    image = Image.open(image)
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Streamlit UI
st.title("Neural Style Transfer App")

# File uploader allows user to add their own images
content_file = st.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])

# Hyperparameters settings
st.sidebar.header("Hyperparameters")
learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
total_steps = st.sidebar.slider("Number of Steps", min_value=1000, max_value=10000, value=6000, step=500)
alpha = st.sidebar.number_input("Alpha", min_value=0.1, max_value=10.0, value=1.0)
beta = st.sidebar.number_input("Beta", min_value=0.01, max_value=1.0, value=0.01)

if st.button('Start Style Transfer'):
    if content_file is not None and style_file is not None:
        # Load images
        content = load_image(content_file)
        style = load_image(style_file)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and generated image
        model = VGG().to(device).eval()
        generated = content.clone().requires_grad_(True)
        optimizer = optim.Adam([generated], lr=learning_rate)

        # Style transfer
        for step in range(total_steps):
            generated_features = model(generated)
            original_img_features = model(content)
            style_features = model(style)
            
            style_loss = original_loss = 0
            for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
                batch_size, channel, height, width = gen_feature.shape
                original_loss += torch.mean((gen_feature - orig_feature) ** 2)
                
                G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
                A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
                style_loss += torch.mean((G - A) ** 2)

            total_loss = alpha * original_loss + beta * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 200 == 0:
                st.write(f"Step {step}, Total Loss: {total_loss.item()}")

        # Save and display the image
        save_image(generated, "generated.png")
        st.image("generated.png", caption="Generated Image", use_column_width=True)
