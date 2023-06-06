
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
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.MaxPool2d(2),

      nn.Conv2d(16, 32, 3, 1, padding=1, bias=True),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.MaxPool2d(2),

      nn.Conv2d(32, 64, 3, 1, padding=1, bias=True),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.MaxPool2d(2),

      nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),

      nn.ConvTranspose2d(32, 16, 3, 2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),

      nn.ConvTranspose2d(16, 8, 3, 2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),

      nn.Conv2d(8, 3, 3, 1, padding=1, bias=True),
      nn.Sigmoid()
    )

  def forward(self, input):
    return self.main(input)
  
#Generato with skip connections
class GeneratorSC(nn.Module):
    def __init__(self, ngpu):
        super(GeneratorSC, self).__init__()
        self.ngpu = ngpu

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.upBlock_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upBlock_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upBlock_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 8, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # self.add_1 = nn.Add()
        # self.add_2 = nn.Add()
        # self.add_3 = nn.Add()

        self.last_conv = nn.Conv2d(8, 3, 3, 1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def main(self, input):
        x1 = self.block_1(input)
        x2 = self.block_2(x1)
        x = self.block_3(x2)

        x = self.upBlock_1(x)
        x = torch.add(x,x2)

        x = self.upBlock_2(x)
        x = torch.add(x,x1)

        x = self.upBlock_3(x)
        
        x = self.last_conv(x)
        x = self.sigmoid(x)
        return x

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
    
#import torch
class VisionTransformer(nn.Module):
    def __init__(self, ngpu, image_size, patch_size, num_channels, emb_dim, num_heads, num_layers, hidden_dim=2048):
        super(VisionTransformer, self).__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.ngpu = ngpu
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.emb_dim = emb_dim

        self.patch_embedding = nn.Conv2d(num_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, num_heads, hidden_dim),
            num_layers
        )

        self.upBlock_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8, 8, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upBlock_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8, 4, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upBlock_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(4, 3, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.last_conv = nn.Conv2d(3, 3, 3, 1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # self.decoder = nn.Sequential(
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, num_channels * patch_size ** 2),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # print(x.shape)
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        # print(x.shape)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)

        x += self.position_embedding
        x = self.transformer_encoder(x)
        # print(x.shape)

        x = x[:, 1:]
        # print(x.shape)
        x = x.view(batch_size, 8, 32, 32)
        # print(x.shape)
        # x = self.decoder(x
        # x = torch.moveaxis(x, 1, )
        # print(x.shape)
        
        x = self.upBlock_1(x)
        x = self.upBlock_2(x)
        x = self.upBlock_3(x)

        x = self.last_conv(x)
        x = self.sigmoid(x)

        # x = x.transpose(1, 2)
        # print(x.shape)
        # x = x.reshape(batch_size, 3, self.image_size,self.image_size)
        # print(x.shape)
        
        return x
