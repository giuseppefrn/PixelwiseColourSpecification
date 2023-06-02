#Test of using a simple autoencoder

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
# from torcheval.metrics import Mean
from torchmetrics import MeanMetric

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
from utils.transforms import *

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

args = {'loss': {"bce": nn.BCELoss(), "mse":nn.MSELoss()} }
    
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
    parser.add_argument('--alpha', type=float, default=1., help='alpha coef for magnitude weight')
    parser.add_argument('--loss_fn', type=str, default='mse', help='loss fn name (mse or bce)')
    parser.add_argument('--grad_reg', action="store_true", help='grandient reg flag')
    parser.add_argument('--run_name', type=str, default='run', help='name for the run')
    parser.add_argument('--test_on', required=True, type=str, choices=["color", "subcolor", "shape"], help="Criterion to use to split test set")
    parser.add_argument('--test_value', required=True, type=str, help="Value to split the test set")
    
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

    data_path = opt.data_dir #'/scratch/gfurnari/transparent/D65'
    label_path = opt.label_dir #'/scratch/gfurnari/transparent/SHADE'
    output_dir = opt.output_dir #'/scratch/gfurnari/transparent-output'

    alpha = torch.tensor(opt.alpha, dtype=torch.float)

    if opt.test_on and opt.test_on == "subcolor":
       opt.test_value = int(opt.test_value)

    ## CREATING OUTPUT DIR
    final_output_dir = os.path.join(output_dir, opt.run_name)
    i = 0

    while os.path.exists(final_output_dir):
        i += 1
        final_output_dir = os.path.join(output_dir, opt.run_name + str(i))

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
          'loss_fn':opt.loss_fn,
          'grad_reg':opt.grad_reg,
          'name': opt.experiment_name,
          'alpha': opt.alpha,
          'test_on': opt.test_on,
          'test_value': opt.test_value
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
    # remove_z_depth(data_path)
    # remove_z_depth(opt.label_dir)

    annotations, test_annotations = build_annoations(data_path, opt.test_on, opt.test_value)
    print("Annotation written", len(annotations))
    print("Test annotation written", len(test_annotations))

    os.makedirs(output_dir, exist_ok=True)
    annotations.to_csv(os.path.join(output_dir,'annotations.csv'),index=False)
    test_annotations.to_csv(os.path.join(output_dir,'test_annotations.csv'),index=False)

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
    splitted = torch.utils.data.random_split(training_data, [0.8, 0.2])

    train_dataloader = DataLoader(splitted[0], batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(splitted[1], batch_size=batch_size, shuffle=True)

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

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    # netD.apply(weights_init)
    
    # Initialize Loss function
    criterion = args['loss'][opt.loss_fn]

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    real_batch = next(iter(train_dataloader))
    
    fixed_noise = real_batch[0].to(device) # fixed images

    # Setup Adam optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Lists to keep track of progress
    img_list = []

    val_deltas = []
    val_losses = []

    avg_losses = []
    val_avg_losses = []

    metric_loss = MeanMetric()
    val_metric_loss = MeanMetric()
    val_metric_deltae = MeanMetric()

    os.makedirs(os.path.join(output_dir, 'generated_albedo'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        
        ############################
        ## TRAIN
        ############################
        netG.train()
        for i, data in enumerate(train_dataloader, 0):
            # Format batch
            netG.zero_grad()
            disc_data = data[0]
            gen_data = disc_data
            
            real_alb = data[1].to(device)
            real_data = disc_data.to(device)       

            b_size = real_alb.size(0)
            outputs = netG(real_data)

            # Calculate the error
            err = criterion(outputs, real_alb)

            with torch.no_grad():
                img_magnitude = calculate_img_gradient(outputs)
                magnitude = torch.mean(img_magnitude)

            if opt.grad_reg:
                total_err = err + alpha * magnitude
            else:
                total_err = err
            
            # Calculate the gradients for this batch
            total_err.backward()
            optimizerG.step()
                        
            metric_loss.update(torch.tensor(total_err.item()))

            # Output training stats
            #TODO move to epoch
            # if i % 100 == 0:
                
            #     print('[%d/%d][%d/%d]\tLoss: %.4f\tTotal Loss: %.4f\tmMagnitude: %.4f'
            #         % (epoch, num_epochs, i, len(train_dataloader),
            #             err.item(), total_err.item(), magnitude))
        
        #end epoch
        #save and reset metrics
        epoch_loss = metric_loss.compute().item()
        metric_loss.reset()
        avg_losses.append(epoch_loss)

        print('[%d/%d]\tEpoch Training Avg Loss: %.4f'
                    % (epoch+1, num_epochs, epoch_loss))
        
        if (epoch % 5 == 0 or epoch == num_epochs-1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True)
                
                fig = plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.imshow(np.transpose(img,(1,2,0)))
                plt.title("Generated Albedo")
                plt.savefig(os.path.join(output_dir, 'generated_albedo', 'gen-albedo-{}'.format(epoch)))
                plt.show()
                plt.close('all')

        ############################
        ##  VALIDATION
        ############################

        netG.eval()
        for i, data in enumerate(valid_dataloader, 0):
            with torch.no_grad():
                gen_data = data[0]
                
                real_alb = data[1].to(device)
                real_data = gen_data.to(device)       

                b_size = real_alb.size(0)

                outputs = netG(real_data)

                # Calculate the error
                err = criterion(outputs, real_alb)

                if opt.grad_reg:
                    img_magnitude = calculate_img_gradient(outputs)
                    magnitude = torch.mean(img_magnitude)

                    total_err = err + alpha * magnitude
                else:
                    total_err = err
                
                val_metric_loss.update(torch.tensor(total_err.item()))
            
                deltae = deltae_2000(real_alb, outputs)
                val_metric_deltae.update(torch.tensor(deltae))
        
        val_loss = val_metric_loss.compute().item()
        val_deltae = val_metric_deltae.compute().item()

        val_metric_loss.reset()
        val_metric_deltae.reset()
            
        if len(val_deltas) > 0 and val_deltae <= min(val_deltas):
            print("\tNew best model!")
            torch.save(netG.state_dict(), os.path.join(output_dir, 'models' ,'model-best.pt'))
            fake = netG(fixed_noise).detach().cpu()
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
            plt.title("Generated Albedo")
            plt.savefig(os.path.join(output_dir, 'albedo-best-gen'))
            plt.show()
            plt.close()
        
        val_deltas.append(val_deltae)
        val_losses.append(val_loss)
        
        print('\tValidation avg loss: %.4f\tDeltaE00: %.4f' % (val_loss, val_deltae))
        
    torch.save(netG.state_dict(), os.path.join(output_dir, 'models' ,'model-last.pt'))
    history = pd.DataFrame({"AvgLoss": avg_losses, "ValLoss": val_losses, "DeltaE": val_deltas})
    history.to_csv(os.path.join(output_dir, 'history.csv'))

    ##TODO test model on test set
    ## free memory 
    del valid_dataloader, train_dataloader, training_data, splitted

    test_data = CustomImageDataset(os.path.join(output_dir,'test_annotations.csv'), transform, transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    test_metric_deltae = MeanMetric()

    netG.eval()
    for i, data in enumerate(test_dataloader, 0):
        with torch.no_grad():
            gen_data = data[0]
                
            real_alb = data[1].to(device)
            real_data = gen_data.to(device)       

            b_size = real_alb.size(0)

            outputs = netG(real_data)

            deltae = deltae_2000(real_alb, outputs)
            test_metric_deltae.update(torch.tensor(deltae))
 
    test_deltae = test_metric_deltae.compute().item()
    print("DeltaE on test set {:.4f}".format(test_deltae))

    fake = outputs.detach().cpu()
    
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
    plt.title("Generated Albedo on test")
    plt.savefig(os.path.join(output_dir, 'albedo-testset'))
    plt.show()
    plt.close()
    
    #PLOT
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    plt.figure(figsize=(10,5))
    plt.title("Loss During Training")
    # plt.plot(G_losses,label="G")
    plt.plot(avg_losses,label="Training")
    plt.plot(val_losses,label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'plots', 'losses'))
    plt.show()
    plt.close('all')

    plt.figure(figsize=(10,5))
    plt.title("Validation $\Delta$E2000 During Training")
    plt.plot(val_deltas)
    plt.xlabel("Epoch")
    plt.ylabel("$\Delta$E2000")
    plt.savefig(os.path.join(output_dir, 'plots', 'deltae2000'))
    plt.show()
    plt.close('all')
    
    #plot some training imgs
    ex = real_batch[1]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Original Shadeless Images")
    plt.imshow(np.transpose(vutils.make_grid(ex, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(output_dir, 'original-albedo'))
    
    ex = real_batch[0]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Original Rendered Images")
    plt.imshow(np.transpose(vutils.make_grid(ex, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(output_dir, 'rendered'))