# Define paths for input and output datasets
TEST_DATASET_HAZY_PATH = 'D:/DL_DL/val/hazy'
TEST_DATASET_OUTPUT_PATH = 'D:/DL_DL/val/new'

# Import necessary libraries
import os
import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Get the list of input images
input_images = os.listdir(TEST_DATASET_HAZY_PATH)
num = len(input_images)
# Initialize list for output images
output_images = []

# Preprocess input and output image paths
for i in range(num):
    output_images.append(os.path.join(TEST_DATASET_OUTPUT_PATH, input_images[i]))
    input_images[i] = os.path.join(TEST_DATASET_HAZY_PATH, input_images[i])
    print(i)


# Define the CycleGAN model architecture
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



# Load the trained model
model_path = 'dl_2_trained_model.pth'  # Update the path accordingly
print("Model Loaded")

# Initialize the model architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator(3, 3, num_blocks=9).to(device)

# Load the pre-trained weights
checkpoint = torch.load(model_path, map_location=device)
print('Pre-Trained Wweights Loaded')

# Load the state dict
model.load_state_dict(checkpoint['G_AB_state_dict'])

# Set the model to evaluation mode
model.eval()
print("Evaluation Mode ON")

# Set up transforms for preprocessing input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Process each image
for i in range(num):
    input_image_path = input_images[i]
    output_image_path = output_images[i]
    print(input_image_path)

    # Perform inference
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        dehazed_image = model(input_tensor).squeeze(0).cpu()

    # Save the dehazed image
    dehazed_image = to_pil_image(dehazed_image)
    dehazed_image.save(output_image_path)
    
print("Dehazing complete.")
