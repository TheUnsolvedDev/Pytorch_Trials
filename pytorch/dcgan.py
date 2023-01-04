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


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.gen_block(z_dim, hidden_dim * 4),
            self.gen_block(hidden_dim * 4, hidden_dim *
                           2, kernel_size=4, stride=1),
            self.gen_block(hidden_dim * 2, hidden_dim),
            self.gen_block(hidden_dim, im_chan,
                           kernel_size=4, final_layer=True),
        )

    def gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


def noise(n_samples, z):
    return torch.randn(n_samples, z).to(device)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.disc_block(im_chan, hidden_dim),
            self.disc_block(hidden_dim, hidden_dim * 2),
            self.disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def disc_loss(gen, disc, criterion, real, num_images, z_dim):
    fake_noise = noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
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

    beta_1 = 0.5
    beta_2 = 0.999
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataloader = DataLoader(
        MNIST('.', download=False, transform=transform),
        batch_size=batch_size,
        shuffle=True)

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(
        disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    gen.train()
    disc.train()
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    for epoch in tqdm(range(epochs)):
        for real, _ in dataloader:
            cur_batch_size = len(real)
            real = real.to(device)

            dloss = disc_loss(gen, disc, criterion, real,
                              cur_batch_size, z_dim)
            disc_opt.zero_grad()
            dloss.backward(retain_graph=True)
            disc_opt.step()

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
                torch.save(gen.state_dict(), 'generator_CNN.pth')
                torch.save(disc.state_dict(), 'discriminator_CNN.pth')


def show_tensor_images(image_tensor, num_images=25):
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def test():
    gen = Generator(64).to(device)
    gen.load_state_dict(torch.load('generator_CNN.pth'))
    fake_noise = noise(25, 64)
    gen.eval()
    with torch.inference_mode():
        fake = gen(fake_noise)

    show_tensor_images(fake)


if __name__ == '__main__':
    model = Generator(z_dim=64)
    torchinfo.summary(model, input_size=(32, 64))

    model = Discriminator()
    torchinfo.summary(model, input_size=(32, 1, 28, 28))
    # train()

    test()
