import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import roc_auc_score
import torch


def print_files(path):
    for current_file in path.iterdir():
        print(current_file)
        
def read_image(path):
    bgr_image = cv2.imread(path)
    # flip image to rgb
    b,g,r = cv2.split(bgr_image)
    rgb_image = cv2.merge([r,g,b])
    return rgb_image
    
def auc_score(y_score,y_true,to_tensor=True):
    score = roc_auc_score(y_true.cpu().numpy(),torch.sigmoid(y_score)[:,1].cpu().numpy())
    if to_tensor:
        score=torch.tensor(score)
    return score