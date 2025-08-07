#This block of code is orginally from WaveGAN paper that publish their work in https://github.com/chrisdonahue/wavegan/blob/master/wavegan.py
import torch
import torch.optim as optim

class WGAN_GP:
    def __init__(self, generator, discriminator, z_dim, lr=0.0001, n_critic=5, lambda_gp=10):
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        epsilon = torch.rand(batch_size, 1, 1).to(real_samples.device)
        interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                       grad_outputs=torch.ones_like(d_interpolates),
                                       create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def train_step(self, real_samples):
        batch_size = real_samples.size(0)

        # Train Discriminator
        self.discriminator.zero_grad()

        z = torch.randn(batch_size, self.z_dim).to(real_samples.device)
        fake_samples = self.generator(z)

        d_real = self.discriminator(real_samples)
        d_fake = self.discriminator(fake_samples.detach())

        gp = self.gradient_penalty(real_samples, fake_samples.detach())

        d_loss = -torch.mean(d_real) + torch.mean(d_fake) + self.lambda_gp * gp
        d_loss.backward()
        self.optimizer_d.step()

        # Train Generator
        if self.n_critic > 0:
            self.generator.zero_grad()

            z = torch.randn(batch_size, self.z_dim).to(real_samples.device)
            fake_samples = self.generator(z)

            d_fake = self.discriminator(fake_samples)

            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            self.optimizer_g.step()

        return d_loss.item(), g_loss.item()