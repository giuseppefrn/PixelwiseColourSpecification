
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

def build_annoations(root_path):
  annotations = pd.DataFrame()
  
  for root, dirs, files in os.walk(root_path):
      for f in files:
        label_root = root.replace('D65', 'SHADE')
        img_path = os.path.join(root,f)
        label_path = os.path.join(label_root,f)

        annotations = annotations.append({0:img_path, 1:label_path}, ignore_index=True)

  return annotations

def build_annoations(root_path, test_on, value):
  annotations = pd.DataFrame()
  test_annotations = pd.DataFrame()
  
  for root, dirs, files in os.walk(root_path):
      for f in files:
        label_root = root.replace('D65', 'SHADE')

        #get info from path
        n = int(root.split('/')[-1]) #subcolor
        c = root.split('/')[-2] #color
        shape = root.split('/')[-3] #shape

        img_path = os.path.join(root,f)
        label_path = os.path.join(label_root,f)

        switch_case = {"color":c, "subcolor":n, "shape":shape}

        #switch case on test_on and value variable
        if test_on:
          if switch_case[test_on] == value:
             test_annotations = test_annotations.append({0:img_path, 1:label_path}, ignore_index=True)
          else:
             annotations = annotations.append({0:img_path, 1:label_path}, ignore_index=True)
                
        else:
          annotations = annotations.append({0:img_path, 1:label_path}, ignore_index=True)
  return annotations, test_annotations

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