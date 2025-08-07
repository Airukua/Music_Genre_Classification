# This code inspired by the implementaion on : https://github.com/UOS-COMP6252/public/blob/main/GANs/cgan.ipynb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_cgan(generator, discriminator, dataloader, num_classes=10, 
               latent_dim=100, epochs=50, device='cuda', save_path=None):
    
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for real_data, labels in loop:
            batch_size = real_data.size(0)

            # Flatten real data
            real_data = real_data.view(batch_size, -1).to(device)
            labels = labels.to(device)

            # Create real and fake labels
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real data
            real_pred = discriminator(real_data, labels)
            real_loss = criterion(real_pred, valid)

            # Fake data
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_data = generator(z, gen_labels)
            fake_pred = discriminator(fake_data.detach(), gen_labels)
            fake_loss = criterion(fake_pred, fake)

            # Total discriminator loss
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            gen_pred = discriminator(fake_data, gen_labels)
            g_loss = criterion(gen_pred, valid)

            g_loss.backward()
            optimizer_G.step()

            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # Save model at each epoch
        if save_path:
            torch.save(generator.state_dict(), f"{save_path}/generator_epoch{epoch+1}.pt")
            torch.save(discriminator.state_dict(), f"{save_path}/discriminator_epoch{epoch+1}.pt")
