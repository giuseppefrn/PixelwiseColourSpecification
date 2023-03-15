
import torch.nn as nn
import torch

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
      nn.Sigmoid()
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
    
class SiameseDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(SiameseDiscriminator, self).__init__()
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
            nn.Linear(128,64)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward_once(self, x):
        output = self.main(x)
        return output

    def forward(self, input1, input2):
        output_1 = self.forward_once(input1)
        output_2 = self.forward_once(input2)

        output = torch.cat((output_1, output_2), 1)
        output = self.fc(output)

        output = self.sigmoid(output)
        return output
    
class SiameseDiscriminatorPixelWise(nn.Module):
    def __init__(self, ngpu):
        super(SiameseDiscriminatorPixelWise, self).__init__()
        self.ngpu = ngpu

        self.encode = nn.Sequential(
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
            nn.Linear(128,64)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Linear(64, 1)
        )

        self.decode = nn.Sequential(
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
            nn.BatchNorm2d(1)
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward_once(self, x):
        output = self.encode(x)
        return output

    def forward(self, input1, input2):
        output_1 = self.forward_once(input1)
        output_2 = self.forward_once(input2)

        output = torch.cat((output_1, output_2), 1)

        output = self.fc(output)
        
        output = self.decode(output)

        output = self.sigmoid(output)
        return output