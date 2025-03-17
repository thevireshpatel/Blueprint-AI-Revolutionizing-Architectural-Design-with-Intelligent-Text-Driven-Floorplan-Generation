import matplotlib.pyplot as plt
import numpy as np
import torch

class TrainingLogger:
    def __init__(self):
        self.losses = {'D': [], 'G': []}

    def log_batch(self, loss_D, loss_G):
        self.losses['D'].append(loss_D)
        self.losses['G'].append(loss_G)

    def log_epoch(self, epoch, generator, device, latent_dim=100, annotation_dim=0):
        noise = torch.randn(1, latent_dim).to(device)
        annotations = torch.zeros(1, annotation_dim).to(device)
        with torch.no_grad():
            generated = generator(noise, annotations)
        
        generated_img = generated.cpu().squeeze().permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(generated_img)
        plt.axis('off')
        plt.title(f"Generated at epoch {epoch}")
        plt.savefig(f"generated_floorplan_epoch_{epoch}.png")
        plt.close()

    def plot_training_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses['D'], label='Discriminator')
        plt.plot(self.losses['G'], label='Generator')
        plt.title('Training Losses')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()