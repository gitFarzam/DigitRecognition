import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from utils import img_label_tolist , csv_writer
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from configuration import data_configuration
from transform import EdgeCrop,SquareCrop
import matplotlib.pyplot as plt

# Check if needed create an annotation file
if data_configuration['new_csv']:
    image_labels = img_label_tolist(path=data_configuration['images_path'] , subset=data_configuration['images_subset'])
    csv_writer(img_label=image_labels)
    print(f"{data_configuration['images_path']} is created by the user!")


df = pd.read_csv(data_configuration['csv_file'])

# creating dataframe and splitting into training and testing set (if required)
if data_configuration['splitting']:
    X_train, X_test, y_train, y_test = train_test_split(df['image'] , df['label'] , test_size=data_configuration['split_ratio'] , random_state=42 , shuffle=True)
    df_train = pd.concat([X_train,y_train] , axis=1).set_index('image').to_csv(data_configuration["csv_train"])
    print(f"{data_configuration['csv_train']} is created by te user!")
    df_test = pd.concat([X_test,y_test] , axis=1).set_index('image').to_csv(data_configuration["csv_test"])
    print(f"{data_configuration['csv_test']} is created by te user!")

# Show dataframe?
if data_configuration['show_df']:
    print(df)

# Show a sample?
if data_configuration['show_sample']: 
    sample = df.iloc[4000]
    print(f"sample image : {sample.image} | sample label : {sample.label}")
    print(f"{data_configuration['images_path']}/{sample.image}")
    im = Image.open(f"{data_configuration['images_path']}/{sample.image}")
    im.show()



class DigitDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx) :
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])


        image = ImageOps.invert(Image.open(img_path).getchannel(channel=3))
         # if we check channels (R=0,G=1,B=2,Alpha=3) , we can see all just have 0 value and just in channel = 3 (alpha) we have value, because samples are gray tranpranet images, they just have a value in alpha channel.
        # and if I just convert the whole image into Gray Scale it will return 0 , so I just get the alpha channel

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    


# build transformer
transform = transforms.Compose([ # the order is important
    
    transforms.ToTensor(), # first convert to tensor
    EdgeCrop(), # cropping edges
    SquareCrop(), # make it like a square
    transforms.Resize(size=28), # resize it to have a same size for all samples
    transforms.ConvertImageDtype(dtype=torch.float32),
])

# Creating training and testing dataset
train_set = DigitDataset(annotations_file=data_configuration['csv_train'] , img_dir=data_configuration['images_path'] , transform=transform)

test_set = DigitDataset(annotations_file=data_configuration['csv_test'] , img_dir=data_configuration['images_path'] , transform=transform)

num_labels = train_set.img_labels['label'].nunique()

# Creating training and testing data loaders
train_loader = DataLoader(dataset=train_set , batch_size=data_configuration['batch_size'] , shuffle=True )
test_loader = DataLoader(dataset= test_set , batch_size = data_configuration['batch_size'] )

#sample
inputs_sample , labels_sample  = next(iter(train_loader))
input_cc = inputs_sample.shape[1]

# print info and samples
if data_configuration['data_info_sample']:
    print(f"train_set len: {len(train_set)} | test_set len: {len(test_set)}")
    print(f"input shape: {inputs_sample.shape} | input sum: {inputs_sample.sum()} | labels shape: {labels_sample.shape}")
    plt.imshow(inputs_sample[0][0] , cmap='gray') ; plt.title('sample') ; plt.show()
    
    to_pil = transforms.ToPILImage(mode='L')
    im = to_pil(inputs_sample[0])
    print(im)
    im.save('data/save/input_sample.png')

