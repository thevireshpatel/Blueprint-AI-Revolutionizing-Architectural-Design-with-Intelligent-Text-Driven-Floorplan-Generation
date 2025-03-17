import torch
import matplotlib.pyplot as plt
from models.generator import Generator

def load_trained_model(generator_path='models/generator.pth', latent_dim=100, annotation_dim=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(latent_dim, annotation_dim).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    return generator

def generate_floorplan(generator, device, latent_dim=100, annotation_dim=0):
    noise = torch.randn(1, latent_dim).to(device)
    annotations = torch.zeros(1, annotation_dim).to(device)
    with torch.no_grad():
        generated = generator(noise, annotations)
    return generated.cpu().squeeze().permute(1, 2, 0).numpy()

def display_floorplan(img, save_path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    latent_dim = 100
    annotation_dim = 10
    
    generator = load_trained_model(latent_dim=latent_dim, annotation_dim=annotation_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(4):
        generated_img = generate_floorplan(generator, device, latent_dim, annotation_dim)
        display_floorplan(generated_img, save_path=f"generated_floorplan_{i}.png")