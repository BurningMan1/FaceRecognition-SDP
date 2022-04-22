
# In[1]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import facenet_pytorch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import numpy as np
import csv
from datetime import datetime
import warnings
import pickle
warnings.filterwarnings("ignore")


# In[2]:

def extract_emb():
    mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()


    # In[3]:


    dataset = datasets.ImageFolder('data2') # photos folder path
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        #print(img.size)
        #Unlike other implementations, calling a facenet-pytorch MTCNN object directly with an image (i.e., using the forward method for those familiar with pytorch) will return torch tensors containing the detected face(s), rather than just the bounding boxes. This is to enable using the module easily as the first stage of a facial recognition pipeline, in which the faces are passed directly to an additional network or algorithm.
        face, prob = mtcnn0(img, return_prob=True)
        boxes, _ = mtcnn0.detect(img)
        #print("boxes:", boxes)
        #print("face: ",face," prob:",prob)
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0))
            #print(emb.shape)
            #print(idx_to_class[idx])
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    # save data
    data = [embedding_list, name_list]
    torch.save(data, 'classifier.pt') # saving my_classifier.pkl file


# In[ ]:




extract_emb()