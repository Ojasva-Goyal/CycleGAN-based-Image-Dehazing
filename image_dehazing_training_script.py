import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        """
        Override the default __getitem__ method to return both the image and its index.
        """
        path, _ = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        # Return the image and its index
        return img, index

import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(os.path.join(root_dir, "hazy"))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "hazy", self.image_filenames[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        # Load corresponding ground truth image
        gt_name = os.path.join(self.root_dir, "GT", self.image_filenames[idx])
        gt_image = Image.open(gt_name)
        if self.transform:
            gt_image = self.transform(gt_image)

        # Return both the hazy image and its ground truth
        return image, gt_image

# Define the CycleGAN model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResidualBlock(256, 256) for _ in range(num_blocks)])
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.res_blocks(out)
        out = self.deconv1(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.tanh(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv5(out)
        return out

# Define the dataset and data loaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomDataset('DL_aug/train', transform=transform)
val_dataset = CustomDataset('DL_aug/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the models
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
G_AB = Generator(3, 3, num_blocks=9).to(device)
G_BA = Generator(3, 3, num_blocks=9).to(device)
D_A = Discriminator(3).to(device)
D_B = Discriminator(3).to(device)

# Define the loss functions and optimizers
criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()

optimizer_G = optim.Adam(
    list(G_AB.parameters()) + list(G_BA.parameters()),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss_D_A = 0.0
    epoch_loss_D_B = 0.0
    epoch_loss_G_A = 0.0
    epoch_loss_G_B = 0.0
    epoch_loss_cycle_A = 0.0
    epoch_loss_cycle_B = 0.0
    for i, (hazy, gt) in enumerate(train_loader):
        hazy = hazy.to(device)  # Move input to the same device as model's weights
        gt = gt.to(device)

        # Update discriminators
        optimizer_D_A.zero_grad()
        real_A = gt
        fake_A = G_BA(hazy)
        real_loss_A = criterion_gan(D_A(real_A), torch.ones_like(D_A(real_A)))
        fake_loss_A = criterion_gan(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
        loss_D_A = (real_loss_A + fake_loss_A) / 2
        loss_D_A.backward()
        optimizer_D_A.step()
        epoch_loss_D_A += loss_D_A.item()

        optimizer_D_B.zero_grad()
        real_B = hazy
        fake_B = G_AB(gt)
        real_loss_B = criterion_gan(D_B(real_B), torch.ones_like(D_B(real_B)))
        fake_loss_B = criterion_gan(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
        loss_D_B = (real_loss_B + fake_loss_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        epoch_loss_D_B += loss_D_B.item()

        # Update generators
        optimizer_G.zero_grad()
        fake_A = G_BA(hazy)
        fake_B = G_AB(gt)
        loss_G_A = criterion_gan(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        loss_G_B = criterion_gan(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        cycle_A = G_AB(fake_B)
        cycle_B = G_BA(fake_A)
        loss_cycle_A = criterion_cycle(cycle_A, gt)
        loss_cycle_B = criterion_cycle(cycle_B, hazy)
        loss_G = loss_G_A + loss_G_B + loss_cycle_A * 10 + loss_cycle_B * 10
        loss_G.backward()
        optimizer_G.step()
        epoch_loss_G_A += loss_G_A.item()
        epoch_loss_G_B += loss_G_B.item()
        epoch_loss_cycle_A += loss_cycle_A.item()
        epoch_loss_cycle_B += loss_cycle_B.item()

    # Compute and print average losses for the epoch
    epoch_loss_D_A /= len(train_loader)
    epoch_loss_D_B /= len(train_loader)
    epoch_loss_G_A /= len(train_loader)
    epoch_loss_G_B /= len(train_loader)
    epoch_loss_cycle_A /= len(train_loader)
    epoch_loss_cycle_B /= len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss_D_A: {epoch_loss_D_A:.4f}, Loss_D_B: {epoch_loss_D_B:.4f}, '
          f'Loss_G_A: {epoch_loss_G_A:.4f}, Loss_G_B: {epoch_loss_G_B:.4f}, '
          f'Loss_Cycle_A: {epoch_loss_cycle_A:.4f}, Loss_Cycle_B: {epoch_loss_cycle_B:.4f}')

    # Save models after every epoch
    save_path = "DL/Bhuman_aug"
    torch.save({
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        'epoch': epoch,
        'loss_D_A': epoch_loss_D_A,
        'loss_D_B': epoch_loss_D_B,
        'loss_G_A': epoch_loss_G_A,
        'loss_G_B': epoch_loss_G_B,
        'loss_cycle_A': epoch_loss_cycle_A,
        'loss_cycle_B': epoch_loss_cycle_B,
    }, f'{save_path}/models_epoch_{epoch+1}.pth')

    # Validation loop
    with torch.no_grad():
        val_loss = 0.0
        for hazy, gt in val_loader:
            hazy = hazy.to(device)
            gt = gt.to(device)

            # Dehaze the images
            dehazed = G_AB(hazy)

            # Compute validation loss
            val_loss += criterion_cycle(dehazed, gt).item() * dehazed.size(0)

        val_loss /= len(val_dataset)

    # Log validation loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

