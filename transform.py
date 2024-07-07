import torch
from torchvision.transforms import Pad

class EdgeCrop():

    def __call__(self, image:torch.Tensor)->torch.Tensor:

        image = image.squeeze(dim=0)
        img_h, img_w = image.shape

        cond = torch.where(image!=1)

        # find the north edge:
        north_edge = torch.where(cond[0] == cond[0].min())[0]
        n = cond[0][north_edge[0]]

        # find the south edge:
        south_edge = torch.where(cond[0] == cond[0].max())[0]
        s = cond[0][south_edge[0]] 

        # find the west edge 
        west_edge = torch.where(cond[1] == cond[1].min())[0]
        w = cond[1][west_edge[0]]

        # find the east edge 
        east_edge = torch.where(cond[1] == cond[1].max())[0]
        e = cond[1][east_edge[0]]

        if s+1 == img_h or e+1 == img_w: # check not to go out of index
            image_cropped = image[n:s , w:e]
        else:
            image_cropped = image[n:s+1 , w:e+1]

        return image_cropped.unsqueeze(dim=0)
    
class SquareCrop():

    def __call__(self, image:torch.Tensor)->torch.Tensor:

        image = image.squeeze(dim=0)
        img_h, img_w = image.shape
        if img_h != img_w:
            max_size = max(img_h,img_w)

            diff = abs(img_h - img_w)
            zero_tensor = torch.ones((max_size,1))

            if img_h > img_w:
                stack = torch.hstack 
            else:
                stack =  torch.vstack 
                zero_tensor = torch.ones((1,max_size))

            for i in range(diff):
                image = stack((zero_tensor,image)) if i%2==0 else stack((image, zero_tensor))
                
        return image.unsqueeze(dim=0)
    


class GrayModifier():

    def __init__(self,threshold):
        self.threshold = threshold

    def __call__(self, image:torch.Tensor)->torch.Tensor:

        image = image.squeeze(dim=0)
        image[image < self.threshold] = 0. # 0. is black
        image[image > self.threshold] = 1. # 1. is white

        return image.unsqueeze(dim=0)
    

class StrokeModifier():

    def __init__(self,shift_r,shift_l):
        self.shift_r = shift_r
        self.shift_l = shift_l

    def __call__(self, image:torch.Tensor)->torch.Tensor:

        image = image.squeeze(dim=0)

        padding = Pad(padding=2*self.shift_r, fill=1 , padding_mode='constant')
        image = (padding(image)).squeeze(0)
        image[image<image.max()*.8] = 0.
        for i in [-self.shift_l,self.shift_r]:
            image = image + torch.roll(image  , shifts=i , dims=0)
            # image[image<image.max()] =0.
        for i in [-self.shift_l,self.shift_r]:
            image = image + torch.roll(image , shifts=i , dims=1)
            # image[image<image.max()] =0.
        
        image[image<image.max()] = image.min()

        return image.unsqueeze(dim=0)
    
