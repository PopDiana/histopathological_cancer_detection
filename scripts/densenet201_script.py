import os

import numpy as np
import pandas as pd 
import numpy as np
from fastai import *
from fastai.vision import *
import torch
import torch.nn as nn
import torchvision
from torchvision.models import *
import cv2
from sklearn.metrics import roc_auc_score

from utils import *

DATA_DIRECTORY = "..." # modify
TRAIN_LABELS =  DATA_DIRECTORY + "/train_labels.csv"
SUBMISSION_LABELS = DATA_DIRECTORY + "/sample_submission.csv"
SIZE = 96
BATCH_SIZE = 64

train_labels = pd.read_csv(TRAIN_LABELS)

transforms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1, max_lighting=0.05, max_warp=0.)

data = ImageDataBunch.from_csv(DATA_DIRECTORY,csv_labels='train_labels.csv',folder='train',
                               ds_tfms=transforms, size=SIZE, suffix='.tif',test='test',bs=BATCH_SIZE)

stats=data.batch_stats()        
data.normalize(stats)

model = cnn_learner(data,densenet201,path='.',metrics=[auc_score], ps=0.8)

model.lr_find()

model.recorder.plot()

learning_rate = 3e-03
model.fit_one_cycle(1,learning_rate)

model.unfreeze()

model.lr_find()

model.recorder.plot()

model.fit_one_cycle(1,slice(1e-6,1e-5))

y_score,y_true = model.get_preds()
prediction_score = auc_score(y_score,y_true)
print(prediction_score)

y_test_score,y_test_true = model.get_preds(ds_type=DatasetType.Test)

submission=pd.read_csv(SUBMISSION_LABELS).set_index('id')

clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])
fname_cleaned=clean_fname(data.test_ds.items).astype(str)

submission.loc[fname_cleaned,'label']=to_np(y_test_score[:,1])
submission.to_csv('submission.csv')



