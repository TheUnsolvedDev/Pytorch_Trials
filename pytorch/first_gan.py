import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchinfo

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


def noise(n_samples, z):
    return torch.randn(n_samples, z).to(device)


class Generator(nn.Module):
    def __init__(self, z=10, im_dim=784, hidden_dim=128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            generator_block(z, hidden_dim),
            generator_block(hidden_dim, hidden_dim*2),
            generator_block(hidden_dim*2, hidden_dim*4),
            nn.Linear(hidden_dim*4, im_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def generator(self):
        return self.model


def discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            discriminator_block(im_dim, hidden_dim),
            discriminator_block(hidden_dim, hidden_dim*2),
            discriminator_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

    def discriminator(self):
        return self.model


def disc_loss(gen, disc, criterion, real, num_images, z_dim):
    fake_noise = noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    disc_fake_loss = criterion(
        disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = disc_real_loss + disc_fake_loss
    return disc_loss/2


def gen_loss(gen, disc, criterion, num_images, z_dim):
    fake_noise = noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


def train(epochs=200):
    criterion = nn.BCEWithLogitsLoss()
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.001
    test_generator = True

    dataloader = DataLoader(
        MNIST('.', download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    gen.train()
    disc.train()
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    for epoch in tqdm(range(epochs)):
        for real, _ in dataloader:
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            dloss = disc_loss(gen, disc, criterion, real,
                              cur_batch_size, z_dim)
            disc_opt.zero_grad()
            dloss.backward()
            disc_opt.step()

            if test_generator:
                old_generator_weights = gen.generator(
                )[0][0].weight.detach().clone()

            gloss = gen_loss(gen, disc, criterion, cur_batch_size, z_dim)
            gen_opt.zero_grad()
            gloss.backward()
            gen_opt.step()

            mean_discriminator_loss += dloss.item() / display_step
            mean_generator_loss += gloss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f" Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1

            if epoch % 50 == 0:
                torch.save(gen.state_dict(), 'generator.pth')
                torch.save(disc.state_dict(), 'discriminator.pth')


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def test():
    gen = Generator(64).to(device)
    gen.load_state_dict(torch.load('generator.pth'))
    fake_noise = noise(25, 64)
    gen.eval()
    with torch.inference_mode():
        fake = gen(fake_noise)

    show_tensor_images(fake)


if __name__ == '__main__':
    model = Generator()
    torchinfo.summary(model, input_size=(32, 10))

    model = Discriminator()
    torchinfo.summary(model, input_size=(32, 784))

    # train()
    test()
