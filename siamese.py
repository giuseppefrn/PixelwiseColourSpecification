import os
import shutil
import numpy as np
import pandas as pd
import argparse 
import torch

import torchvision.transforms.functional as Functional
# import torch.nn._reduction as _Reduction
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from model.model import *
from utils.dataload import build_annoations, CustomImageDataset

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

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, label, output, mask):
        error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
        error = torch.sum(error, dim=(1,2))
        non_zero = torch.count_nonzero(mask, (1,2))

        error = torch.mean(torch.div(error,non_zero))

        return error
    
if __name__ == '__main__':
    print('End')

    parser = argparse.ArgumentParser()
    #TODO add others illuminants
    parser.add_argument('--illuminant', type=str, default='D65', help='illuminant to use [D65, F11, F4, ...], in case of folder_split it can be used to name the output dir')
    parser.add_argument('--data_dir', type=str, default='multiviews', help='path to the dataset root')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--output_dir', type=str, default='experiments', help='output directory pathname')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_views', type=int, default=16, help='number of views to use, 1 for single view model')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--test', type=int, choices=[0,1,2], default=2, help='0 dont test the model, 1 test at the end of training, 2 test after each epoch')
    parser.add_argument('--mode', type=str, default='RGBD', help='RGBD or RGB mode', choices=['RGBD', 'RGB'])
    parser.add_argument('--test_on', type=str, choices=['subcolor', 'color', 'shape', 'folder_split'], default='subcolor', help="Select test set based on different subcolor/color/shape")
    parser.add_argument('--value', help='Value for test selection, can be a color (str), subcolor ([0 - 9]), shape (str)', default=5)

    opt = parser.parse_args()

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
    netD = SiameseDiscriminatorPixelWise(ngpu).to(device)

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
            real_data = data[0].to(device)

            b_size = real_cpu.size(0)
            label = torch.full((b_size,256,256), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu, real_data).view(b_size,256,256)

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
            
            fake = netG(real_data)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach(), real_data).view(b_size,256,256)

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
            output = netD(fake, real_data).view(b_size,256,256)

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