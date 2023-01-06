import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchinfo
import math
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mnist_shape = (1, 28, 28)
num_classes = 10


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()


class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            self.gen_block(input_dim, hidden_dim * 4),
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
        return noise.view(len(noise), self.input_dim, 1, 1)

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


def one_hotted_labels(labels, n_classes):
    return nn.functional.one_hot(labels, n_classes)


def combine_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined


def get_input_dimensions(z_dim, mnist_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def train(epochs=200):
    criterion = nn.BCEWithLogitsLoss()
    z_dim = 64
    batch_size = 128
    display_step = 200
    lr = 0.002

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataloader = DataLoader(
        MNIST('.', download=False, transform=transform),
        batch_size=batch_size,
        shuffle=True)

    generator_input_dim, discriminator_im_chan = get_input_dimensions(
        z_dim, mnist_shape, num_classes)
    gen = Generator(input_dim=generator_input_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator(im_chan=discriminator_im_chan).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    cur_step = 0
    generator_losses = []
    discriminator_losses = []
    noise_and_labels = False
    fake = False

    fake_image_and_labels = False
    real_image_and_labels = False
    disc_fake_pred = False
    disc_real_pred = False

    gen.train()
    disc.train()
    for epoch in tqdm(range(epochs)):
        for real, labels in dataloader:
            cur_batch_size = len(real)
            real = real.to(device)

            one_hot_labels = one_hotted_labels(labels.to(device), num_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(
                1, 1, mnist_shape[1], mnist_shape[2])

            fake_noise = noise(cur_batch_size, z_dim)
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels.detach())
            disc_real_pred = disc(real_image_and_labels)

            disc_opt.zero_grad()
            disc_fake_loss = criterion(
                disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(
                disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()
            discriminator_losses += [disc_loss.item()]

            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            gen_opt.zero_grad()
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(
                disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            generator_losses += [gen_loss.item()]

            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(
                    discriminator_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            elif cur_step == 0:
                print("Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
            cur_step += 1

        if epoch % 50 == 0:
            torch.save(gen.state_dict(), 'generator_conditional_CNN.pth')
            torch.save(disc.state_dict(), 'discriminator_conditional_CNN.pth')


def interpolate_class(first_number, second_number):
    first_label = one_hotted_labels(
        torch.Tensor([first_number]).long(), num_classes)
    second_label = one_hotted_labels(
        torch.Tensor([second_number]).long(), num_classes)

    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
    interpolation_labels = first_label * \
        (1 - percent_second_label) + second_label * percent_second_label

    noise_and_labels = combine_vectors(
        interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation,
                       nrow=int(math.sqrt(n_interpolation)), show=False)


if __name__ == '__main__':
    # train()
    ### Change me! ###
    start_plot_number = 1  # Choose the start digit
    ### Change me! ###
    end_plot_number = 5  # Choose the end digit
    # Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)
    n_interpolation = 9
    interpolation_noise = noise(
        1, z=64).repeat(n_interpolation, 1)

    generator_input_dim, discriminator_im_chan = get_input_dimensions(
        64, mnist_shape, num_classes)
    gen = Generator(input_dim=generator_input_dim).to(device)
    gen.load_state_dict(torch.load('generator_conditional_CNN.pth'))
    gen.eval()

    plt.figure(figsize=(8, 8))
    interpolate_class(start_plot_number, end_plot_number)
    _ = plt.axis('off')

    plot_numbers = [2, 3, 4, 5, 7]
    n_numbers = len(plot_numbers)
    plt.figure(figsize=(8, 8))
    for i, first_plot_number in enumerate(plot_numbers):
        for j, second_plot_number in enumerate(plot_numbers):
            plt.subplot(n_numbers, n_numbers, i * n_numbers + j + 1)
            interpolate_class(first_plot_number, second_plot_number)
            plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0)
    plt.show()
    plt.close()
