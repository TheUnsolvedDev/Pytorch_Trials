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


class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
            )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - \
        torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss


def train(epochs=200):
    n_epochs = 100
    z_dim = 64
    display_step = 50
    batch_size = 128
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    c_lambda = 10
    crit_repeats = 5
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
    crit = Critic().to(device)
    crit_opt = torch.optim.Adam(
        crit.parameters(), lr=lr, betas=(beta_1, beta_2))
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

    # gen.load_state_dict(torch.load('generator_WP_CNN.pth'))
    # crit.load_state_dict(torch.load('critic_WP_CNN.pth'))

    gen.train()
    crit.train()
    cur_step = 0
    generator_losses = []
    critic_losses = []

    for epoch in tqdm(range(epochs+1)):
        for real, _ in dataloader:
            cur_batch_size = len(real)
            real = real.to(device)

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                crit_opt.zero_grad()
                fake_noise = noise(cur_batch_size, z_dim)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1,
                                     device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(
                    crit_fake_pred, crit_real_pred, gp, c_lambda)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                crit_loss.backward(retain_graph=True)
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]

            gen_opt.zero_grad()
            fake_noise_2 = noise(cur_batch_size, z_dim)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()
            gen_opt.step()

            generator_losses += [gen_loss.item()]
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(
                    f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")

            cur_step += 1

            if epoch % 50 == 0:
                torch.save(gen.state_dict(), 'generator_WP_CNN.pth')
                torch.save(crit.state_dict(), 'critic_WP_CNN.pth')
    step_bins = 20
    num_examples = (len(generator_losses) // step_bins) * step_bins
    plt.plot(
        range(num_examples // step_bins),
        torch.Tensor(
            generator_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Generator Loss"
    )
    plt.plot(
        range(num_examples // step_bins),
        torch.Tensor(
            critic_losses[:num_examples]).view(-1, step_bins).mean(1),
        label="Critic Loss"
    )
    plt.legend()
    plt.show()


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

    model = Critic()
    torchinfo.summary(model, input_size=(32, 1, 28, 28))
    train()

    test()
