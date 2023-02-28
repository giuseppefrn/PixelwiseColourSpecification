import os
import shutil
import numpy as np
import pandas as pd
import torch

import torchvision.transforms.functional as Functional
# import torch.nn._reduction as _Reduction
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms


import argparse 
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils


def remove_z_depth(path):
    # remove all Z_depth images
    root_path = path

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if 'Z_depth' in file:
             os.remove(os.path.join(root, file))

def copy_folder_content(src_path, dest_path):
    # construct the src path and file name
    files = os.listdir(src_path)

    os.makedirs(dest_path, exist_ok=True)

    for f in files:
      output_fname = f
      if os.path.exists(os.path.join(dest_path, output_fname)):
        i = len(files)
        while os.path.exists(os.path.join(dest_path, output_fname)):
          output_fname = f.split('.')[0][:-4] #take image name
          output_fname = output_fname + '-' + str(i).zfill(4) + '.png'
          i += 1
      shutil.copyfile(os.path.join(src_path, f), os.path.join(dest_path, output_fname))
      
def copy_imgs(root_path, output_path):
  for root, dirs, _ in os.walk(root_path):
    for dir in dirs:
      if dir in [str(i) for i in range(10)]:
        print('copying', os.path.join(root,dir))
        copy_folder_content(os.path.join(root,dir), output_path)

def build_annoations(root_path):
  annotations = pd.DataFrame()
  
  for root, dirs, files in os.walk(root_path):
      for f in files:
        label_root = root.replace('D65', 'SHADE')
        img_path = os.path.join(root,f)
        label_path = os.path.join(label_root,f)

        annotations = annotations.append({0:img_path, 1:label_path}, ignore_index=True)

  return annotations

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx, 0])
        image = read_image(img_path)        

        label_path = os.path.join(self.img_labels.iloc[idx, 1])
        label = read_image(label_path)

        ### TEST MASK ####
        mask = torch.clone(label[0] + label[1] + label[2])
        # mask[mask == 2] = 0
        mask[mask < 5] = 0
        mask[mask >= 5] = 1


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, mask

def build_annoations_multiview(root_path):
  annotations = pd.DataFrame()

  for shape in os.listdir(root_path):
    shape_path = os.path.join(root_path, shape)
    for color in os.listdir(shape_path):
      color_path = os.path.join(shape_path, color)
      for subcolor in os.listdir(color_path):
        #path to images dir
        subcolor_path = os.path.join(color_path, subcolor)
        label_root = subcolor_path.replace('D65', 'SHADE')

        annotations = annotations.append({0:subcolor_path, 1:label_root}, ignore_index=True)
  return annotations


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = Functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class CustomImageDatasetMultiView(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
      self.img_labels = pd.read_csv(annotations_file)
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):
      return len(self.img_labels)

    def __getitem__(self, idx):
      images_path = self.img_labels.iloc[idx, 0]
      images_list = os.listdir(images_path)
      
      images = [read_image(os.path.join(images_path, images_list[i])) for i in range(len(images_list))]
      
      labels_path = self.img_labels.iloc[idx, 1]
      labels_list = os.listdir(labels_path)
      labels = [read_image(os.path.join(labels_path, labels_list[i])) for i in range(len(labels_list))]

      if self.transform:
          images = [self.transform(images[i]) for i in range(len(images))]
      if self.target_transform:
          labels = [self.target_transform(labels[i]) for i in range(len(labels))]
      return images, labels

#Generator Code 
class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      nn.Conv2d(3, 16, 3, 1, padding=1, bias=True),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.MaxPool2d(2),

      nn.Conv2d(16, 32, 3, 1, padding=1, bias=True),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.MaxPool2d(2),

      nn.Conv2d(32, 64, 3, 1, padding=1, bias=True),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.MaxPool2d(2),

      nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(True),

      nn.ConvTranspose2d(32, 16, 3, 2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU(True),

      nn.ConvTranspose2d(16, 8, 3, 2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU(True),

      nn.Conv2d(8, 3, 3, 1, padding=1, bias=True),
    )

  def forward(self, input):
    return self.main(input)

# Discriminator code

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(16),

            nn.Flatten(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DiscriminatorPixelWise(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorPixelWise, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(16),

            #latent 

            nn.Flatten(),
            nn.Linear(128,64),
            # nn.Sigmoid(),

            #reshape
            nn.Unflatten(-1,(1,8,8)),

            nn.ConvTranspose2d(1, 1, 3, 2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(1, 1, 3, 2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(1, 1, 3, 2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(1, 1, 3, 2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(1, 1, 3, 2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, label, output, mask):
        error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
        error = torch.sum(error, dim=(1,2))
        non_zero = torch.count_nonzero(mask, (1,2))

        error = torch.mean(torch.div(error,non_zero))

        return error

# def custom_loss(label, output, mask):
#     # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='mean')
#     # summed_by_img = torch.sum(error, dim=(1,2))
#     # non_zero = torch.count_nonzero(mask, (1,2))
#     # final_err = torch.mean(summed_by_img/non_zero)
#     return nn.functional.binary_cross_entropy(label, output, reduction='mean')

if __name__ == '__main__':
    print('End')

    ## ARGUMENTS ##
    # Root directory for dataset
    dataroot = "/content/imgs"
    # Number of workers for dataloader
    workers = 2
    # Batch size during training
    batch_size = 64
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 100
    # Learning rate for optimizers
    lr = 0.001
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    data_path = '/scratch/gfurnari/transparent/D65'
    label_path = '/scratch/gfurnari/transparent/SHADE'
    output_dir = '/scratch/gfurnari/transparent-output'

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #run just once
    # remove_z_depth('/scratch/gfurnari/transparent/')

    annotations = build_annoations(data_path)
    print(len(annotations))
    annotations.head()

    os.makedirs(output_dir, exist_ok=True)
    annotations.to_csv(os.path.join(output_dir,'annotations.csv'),index=False)

    image_size = 256 

    transform = transforms.Compose([
                                transforms.Resize(image_size),
                                #  transforms.ToTensor(),
                                #  transforms.ConvertDtype(torch.float),
                                    transforms.ConvertImageDtype(torch.float),
                                    # normalize_images
                                #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
                                ])
    
    training_data = CustomImageDataset(os.path.join(output_dir,'annotations.csv'), transform, transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    train_features, labels, masks = next(iter(train_dataloader))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = DiscriminatorPixelWise(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD) 
    
    # Initialize BCELoss function
    criterion = CustomLoss()
    # custom_loss = CustomBCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator

    real_batch = next(iter(train_dataloader))
    fixed_noise = real_batch[0].to(device) # fixed images

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[1].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,256,256), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(b_size,256,256)

            ## add mask
            mask = data[2].to(device)
            # output = torch.clamp(output + mask, max=1) # no need for true label (since all is 1)

            # Calculate loss on all-real batch
            # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
            # summed_by_img = torch.sum(error, dim=(1,2))
            # non_zero = torch.count_nonzero(mask, (1,2))
            # errD_real = torch.mean(summed_by_img/non_zero)

            errD_real = criterion(output, label, mask)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            real_data = data[0].to(device)
            fake = netG(real_data)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(b_size,256,256)

            # Calculate D's loss on the all-fake batch
            # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
            # summed_by_img = torch.sum(error, dim=(1,2))
            # non_zero = torch.count_nonzero(mask, (1,2))
            # errD_fake = torch.mean(summed_by_img/non_zero)

            errD_fake = criterion(output, label, mask)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            # add mask to labels
            # label = torch.clamp(label + mask, max=1) # no need all is already 1

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(b_size,256,256)

            # Calculate G's loss based on this output
            errG = criterion(output, label, mask)

            # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
            # summed_by_img = torch.sum(error, dim=(1,2))
            # non_zero = torch.count_nonzero(mask, (1,2))
            # errG = torch.mean(summed_by_img/non_zero)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    #PLOT
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'losses'))
    plt.show()

    for i in range(len(img_list)):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(img_list[i],(1,2,0)))
        plt.title("Generated Albedo")
        plt.savefig(os.path.join(output_dir, 'gen-albedo-{}'.format(i)))
        plt.show()


    ex = real_batch[1]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Original Shadeless Images")
    plt.imshow(np.transpose(vutils.make_grid(ex, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(output_dir, 'original-albedo'))

    # Plot some training images
    ex = real_batch[0]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Original Rendered Images")
    plt.imshow(np.transpose(vutils.make_grid(ex, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(output_dir, 'rendered'))