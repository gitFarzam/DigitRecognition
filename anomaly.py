import torch

def white_detector(image:torch.Tensor)-> bool:
    w_mean_th = 0.995
    if image.mean() < w_mean_th:
        return True

def black_detector(image:torch.Tensor)-> bool:
    b_mean_th = 0.55
    if image.mean() > b_mean_th:
        return True
