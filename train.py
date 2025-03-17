import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import AnnotatedFloorplanDataset
from utils.training_utils import TrainingLogger

def train_gan(images_dir, annotations_path, epochs=50, batch_size=32, latent_dim=100, annotation_dim=0):
    dataset = AnnotatedFloorplanDataset(images_dir, annotations_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    generator = Generator(latent_dim, annotation_dim).to(device)
    discriminator = Discriminator(annotation_dim).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    adversarial_loss = torch.nn.BCELoss()
    
    logger = TrainingLogger()
    
    for epoch in range(epochs):
        for i, (real_imgs, annotations) in enumerate(dataloader):
            real_labels = torch.ones(real_imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(real_imgs.size(0), 1).to(device)
            
            real_imgs = real_imgs.to(device)
            annotations = annotations.to(device) if annotation_dim > 0 else None
            
            # Train Discriminator
            d_real = discriminator(real_imgs, annotations)
            d_real_loss = adversarial_loss(d_real, real_labels)
            
            noise = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(noise, annotations)
            d_fake = discriminator(fake_imgs.detach(), annotations)
            d_fake_loss = adversarial_loss(d_fake, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            noise = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(noise, annotations)
            d_fake = discriminator(fake_imgs, annotations)
            g_loss = adversarial_loss(d_fake, real_labels)
            
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            
            logger.log_batch(loss_D=d_loss.item(), loss_G=g_loss.item())
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} \
                      Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
        
        logger.log_epoch(epoch, generator, device, latent_dim, annotation_dim)
    
    os.makedirs('models', exist_ok=True)
    torch.save(generator.state_dict(), 'models/generator.pth')
    print("Training complete. Models saved to 'models/' directory.")
    
    logger.plot_training_curves()

if __name__ == "__main__":
    images_dir = "data/floorplan_images"
    annotations_path = "data/annotations.pkl"
    annotation_dim = 10
    
    train_gan(images_dir, annotations_path, annotation_dim=annotation_dim)