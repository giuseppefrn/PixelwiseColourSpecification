import os
import shutil
import numpy as np
import pandas as pd
import argparse 
import torch
import yaml

import torchvision.transforms.functional as Functional
# import torch.nn._reduction as _Reduction
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def calculate_img_gradient(inputs):
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]

    channels = inputs.size()[1]
    sobel_y_kernel = torch.tensor(sobel_y, dtype=torch.float).unsqueeze(0).expand(1, channels, 3, 3).to(device)
    sobel_x_kernel = torch.tensor(sobel_x, dtype=torch.float).unsqueeze(0).expand(1, channels, 3, 3).to(device)

    malignacy_y = nn.functional.conv2d(inputs, sobel_y_kernel, stride=1, padding=1,)
    malignacy_x = nn.functional.conv2d(inputs, sobel_x_kernel, stride=1, padding=1,)

    magnitude = torch.sqrt((torch.square(malignacy_x)) + torch.square(malignacy_y))
    # orientation = torch.arctan2(malignacy_y, malignacy_x) * (180 / np.pi) % 180

    return magnitude
    
if __name__ == '__main__':
    print('Start')
    parser = argparse.ArgumentParser()
    #TODO add others illuminants
    parser.add_argument('--data_dir', type=str, default='/scratch/gfurnari/transparent/D65', help='path to the dataset')
    parser.add_argument('--label_dir', type=str, default='/scratch/gfurnari/transparent/SHADE', help='path to the labels')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--output_dir', type=str, default='/scratch/gfurnari/outputs', help='output directory pathname')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--betal', type=float, default=0.5, help='beta1 value')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 value')
    parser.add_argument('--workers', help='Number of workers for dataloader', default=1)
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus')
    parser.add_argument('--experiment_name', type=str, default='None', help='experiment name')
    # parser.add_argument('--noise', type=bool, default=False, help='boolean add or not the noise to discr inputs')
    parser.add_argument('--noise_mean', type=float, default=0, help='Mean of the gaussian distribution to generate noise')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Dev std of the gaussian distribution to generate noise')
    parser.add_argument('--split_train', type=int, default=0, help='To split training differently for Gen and Disc ')
    parser.add_argument('--add_noise', type=int, default=0, help='if use or not instance noise')
    parser.add_argument('--alpha', type=int, default=1, help='alpha coef for magnitude weight')
    
    opt = parser.parse_args()

    ## ARGUMENTS ##
    # Root directory for dataset
    dataroot = "/content/imgs"
    # Number of workers for dataloader
    workers = opt.workers
    # Batch size during training
    batch_size = opt.batch
    # Number of training epochs
    num_epochs = opt.epochs
    # Learning rate for optimizers
    lr = opt.lr
    # Beta1 hyperparam for Adam optimizers
    beta1 = opt.betal
    beta2 = opt.beta2
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = opt.ngpu

    noise_mean = opt.noise_mean
    noise_std = opt.noise_std

    data_path = opt.data_dir #'/scratch/gfurnari/transparent/D65'
    label_path = opt.label_dir #'/scratch/gfurnari/transparent/SHADE'
    output_dir = opt.output_dir #'/scratch/gfurnari/transparent-output'

    different_split = bool(opt.split_train)
    add_noise = bool(opt.add_noise)

    alpha = opt.alpha

    ## CREATING OUTPUT DIR

    final_output_dir = os.path.join(output_dir, 'run')
    i = 0

    while os.path.exists(final_output_dir):
        i += 1
        final_output_dir = os.path.join(output_dir, 'run' + str(i))
        # run_list = os.listdir(output_dir)
        # i = len(run_list)
        # final_output_dir = os.path.join(output_dir, 'run' + str(i))
        # print('Eperiment folder already exists - creating: {}'.format(final_output_dir))

    print('Experiments directory:', final_output_dir)
    os.makedirs(final_output_dir ,exist_ok=True)

    with open(os.path.join(final_output_dir, 'configuration.yaml'), 'w') as f:
      yaml.dump(
        {
          'data_dir':opt.data_dir,
          'batch':opt.batch,
          'output_dir':final_output_dir,
          'epochs': opt.epochs,
          'lr': lr,
          'beta1': beta1,
          'beta2':beta2,
          'noise_m':noise_mean,
          'noise_std': noise_std,
          'name': opt.experiment_name
        }
        , f)

    output_dir = final_output_dir

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #run just once
    remove_z_depth(data_path)
    remove_z_depth(opt.label_dir)

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
    
    # data_augm = transforms.Compose([
    #    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    #    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    # ])
    
    training_data = CustomImageDataset(os.path.join(output_dir,'annotations.csv'), transform, transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    train_features, labels, masks = next(iter(train_dataloader))

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)

    # Create the generator
    netG = GeneratorSC(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    # netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = SiameseDiscriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # netD.apply(weights_init)

    # Print the model
    print(netD) 
    
    # Initialize BCELoss function
    # criterion = CustomLoss()
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator

    real_batch = next(iter(train_dataloader))
    
    fixed_noise = real_batch[0].to(device) # fixed images

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    d_res = []

    ### TEST SOBEL
    # sobelX = nn.Conv2d(3, 3, 3, stride=1, padding=0, bias=False, padding_mode='zeros', device=device)
    # sobelX.weight = torch.nn.Parameter(torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],requires_grad=False))
    # sobelY = nn.Conv2d(3, 3, 3, stride=1, padding=0, bias=False, padding_mode='zeros', device=device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_dataloader, 0):
            netD.zero_grad()

            # Format batch
            if different_split:
                disc_data = data[0][0:len(data[0])//2]
                gen_data = data[0][len(data[0])//2:]

                real_alb =  data[1][0:len(data[0])//2].to(device)
                real_data = disc_data.to(device)

                real_alb_g = data[1][len(data[0])//2:].to(device)
                real_data_g = gen_data.to(device)
            
            else:
                disc_data = data[0]
                gen_data = disc_data
                real_alb = data[1].to(device)
                real_alb_g = real_alb
                real_data = disc_data.to(device)
                real_data_g = real_data            

            b_size = real_alb.size(0)
            label = torch.full((b_size,1), real_label, dtype=torch.float, device=device)

            ## WITH NOISE
            if add_noise:
                additive_noise = torch.normal(mean=noise_mean, std=noise_std, size=(b_size,3,256,256)).to(device)
                noisy_label = torch.add(real_alb, additive_noise)
                output = netD(noisy_label, real_data).view(b_size, 1)
            
            ## WITHOUT NOISE
            else:
            ## Forward pass real batch through D
                output = netD(real_alb, real_data).view(b_size, 1)

            if epoch == num_epochs - 1:
              d_res.append(output.detach().cpu())

            # Calculate loss on all-real batch
            # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
            # summed_by_img = torch.sum(error, dim=(1,2))
            # non_zero = torch.count_nonzero(mask, (1,2))
            # errD_real = torch.mean(summed_by_img/non_zero)

            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ############################
            ## Train with all-fake batch
            ############################

            # Generate batch of latent vectors
            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            
            label = torch.full((real_alb_g.size(0), 1), fake_label, dtype=torch.float, device=device)
            fake = netG(real_data_g)

            # label.fill_(fake_label)

            # Classify all fake batch with D

            ## WITH NOISE
            if add_noise:
                additive_noise = torch.normal(mean=noise_mean, std=noise_std, size=(b_size,3,256,256)).to(device)
                noisy_fake = torch.add(fake.detach(), additive_noise)
                output = netD(noisy_fake, real_data_g).view(real_alb_g.size(0), 1)

            ##without NOISE
            else:
                output = netD(fake, real_data_g).view(real_alb_g.size(0), 1)

            if epoch == num_epochs - 1:
              d_res.append(output.detach().cpu())

            # Calculate D's loss on the all-fake batch
            # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
            # summed_by_img = torch.sum(error, dim=(1,2))
            # non_zero = torch.count_nonzero(mask, (1,2))
            # errD_fake = torch.mean(summed_by_img/non_zero)

            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            label_2= torch.full((real_alb_g.size(0),1), real_label, dtype=torch.float, device=device)
            label = torch.full((b_size,1), real_label, dtype=torch.float, device=device)

            netG.zero_grad()
            # label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D

            # generate the first half of fakes
            fake_2 = netG(real_data)
            output_2 = netD(fake_2, real_data).view(b_size, 1)
            errG_2 = criterion(output_2, label_2)

            ### TEST SOBEL
            # res_sobel_x = sobelX(fake_2)
            with torch.no_grad():
                img_g_magnitude_1 = calculate_img_gradient(fake)
                img_g_magnitude_2 = calculate_img_gradient(fake_2)

                g_magnitude = 0.5 * (torch.mean(img_g_magnitude_1) + torch.mean(img_g_magnitude_2)) 

            if different_split:
                output = netD(fake, real_data_g).view(b_size, 1)
                errG_1 = criterion(output, label)

                # Calculate G's loss based on this output
                errG = errG_1 + errG_2 + alpha * g_magnitude
            else: errG = errG_2 + g_magnitude

            # error = nn.functional.binary_cross_entropy(label, output, weight=mask, reduction='none')
            # summed_by_img = torch.sum(error, dim=(1,2))
            # non_zero = torch.count_nonzero(mask, (1,2))
            # errG = torch.mean(summed_by_img/non_zero)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            if len(G_losses) > 0 and errG.item() < min(G_losses):
               print('New best Generator!')
               with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    fig = plt.figure(figsize=(8,8))
                    plt.axis("off")
                    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
                    plt.title("Generated Albedo")
                    plt.savefig(os.path.join(output_dir, 'albedo-best-gen'))
                    plt.show()
                    plt.close()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tmMagnitude: %.4f'
                    % (epoch, num_epochs, i, len(train_dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, g_magnitude))

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

    plt.close('all')

    d_res = np.concatenate([x.numpy().flatten() for x in d_res])
    
    print(d_res.shape)
    
    # for x in d_res:
    #    print(x.shape)

    # d_res = np.array(d_res).flatten()
    # print(d_res.shape)

    histplot = sns.histplot(d_res)
    fig = histplot.get_figure()
 
    # use savefig function to save the plot and give
    # a desired name to the plot.
    fig.savefig(os.path.join(output_dir, 'discriminator-predictions.png'))