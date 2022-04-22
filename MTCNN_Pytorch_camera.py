
# In[1]:
import keyboard
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
#import mmcv

# In[2]:


mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
# We use mtcnn to detect multiple faces in our cam
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
# This return a pretrained model that is vggface2
resnet = InceptionResnetV1(pretrained='vggface2').eval()
time_before_last_save = time.time();

# In[ ]:


def create_detector():
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
    return mtcnn


def create_recognizer():
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return resnet


# In[3]:


frame_counter_rate = 5


# In[4]:


def load_embeddings(filename='classifier.pt'):
    """The function load preprocessed embeddings from file
        
    The function load preprocessed Torch tensor from .pt(pytorch format)
    which contains tensors itself and the name of the subject
    Parameters
    ----------
    filename - name of the file generated with Embedding extraction
    ----------
    Returns
    ----------
    name_list - names of all subjects from dataset
    embedding_list - embeddings of the subjects
    ----------
       """
    load_data = torch.load(filename)
    embedding_list = load_data[0]
    name_list = load_data[1]
    return embedding_list, name_list


# In[5]:


def check_opening(cam):
    """Function to check that camera can grab the video
        
    The function try to connect to the camera. If cannot connect
    it will to the console and wait 1 second
    
    Parameters
    ----------
    cam - cv2.VideoCapture object, which try to connect to specified IP Camera
    ----------
    """
    while not cam.isOpened():
        cam = cv2.VideoCapture('<connection_link>')
        cv2.waitKey(1000)
        print("Wait for the header")


# In[6]:


def process_frame_or_not(frame_counter):
    """This function will make to process only every frame_counter_rate frame
        
    The function will check if the number of frame is divisible by frame_counter_rate
    and will give it a try to process frame and find the faces.
    Parameters
    ----------
    frame_counter - which frame is processed
    ----------
    
    Returns
    ----------
    True if divisible
    False if not
    """
    global frame_counter_rate
    if frame_counter % frame_counter_rate == 0:
        return True
    return False


# In[7]:


def process_frame(cam):
    """The function preprocess frame
        
    Function will resize object by reducing the size of frame
    for faster inference
    Parameters
    ----------
    cam - cv2.VideoCapture object, which connected to specified IP Camera
    ----------
    
    Returns
    ----------
    ret - variable to check if opencv could grab the image
    frame - preprocessed frame for inference
    ----------
    """
    ret, frame = cam.read()
    cv2.WaitKey(20);
    frame = cv2.resize(frame, (0, 0), interpolation=cv2.INTER_CUBIC)
    return ret, frame


# In[8]:


def fail_to_grab(ret):
    """The function to show will it grab image or not
        
    Function will show if opencv failed to grab picture
    Parameters
    ----------
    ret - variable to check if opencv could grab the image
    ----------
    
    Returns
    ----------
    True if failed
    False if not
    ----------
    """
    if not ret:
        return True
    return False


# In[9]:


def recognize_faces(frame, resnet, mtcnn, embedding_list, name_list):
    """The function will show if it recognized face and draw box
        
    The function create mtcnn object and detect faces, after which
    will pass images through neural network, to extract embeddings.
    Then, comapre every embedding with the embedding which contains inside
    the dataset. 
    If the face is detected, it will draw the box around him, and if the face
    recognized add the text to bounding box, following writing the name and the time of detection to
    logger file.
    
    Parameters
    ----------
    frame - preprocessed frame
    resnet - embedding extractor model
    ----------
    
    Returns
    ----------
    boxes - the coordinates of bounding boxes
    frame - frame with bounding boxes
    ----------
    """
    img = Image.fromarray(frame[1])
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
        # return boxed faces
        for i, prob in enumerate(prob_list):
            if prob > 0.80:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                name = name_list[min_dist_idx]  # get name corrosponding to minimum dist

                original_frame = frame[1].copy()  # storing copy of frame before drawing on it
                frame = draw_box(boxes, frame, name, min_dist, i)
        return boxes, frame
    else:
        return None, frame


# In[10]:


def write_to_csv(filename, data):
    """The function will write the name to .csv file
        
    If the face was recognized, it will write it to csv.logger
    Parameters
    ----------
    filename - csv file, where to write the data
    data - name and time of detection
    ----------
    
    Returns
    ----------
    True if failed
    False if not
    ----------
    """
    with open(filename, 'a', encoding='UTF8', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(data)


# In[11]:
def save_face(frame):
    global time_before_last_save;
    if not os.path.exists('unknownfolderpath'):
        os.mkdir('unknownfolderpath')
    save_time = time.time();
    if(save_time - time_before_last_save >=5.0):
        time_before_last_save = save_time;
        img_name = "unknownfolderpath/{}.jpg".format(int(time.time()))
        cv2.imwrite(img_name, frame[1])
        print(" saved: {}".format(img_name))
    return 1

def draw_box(boxes, frame, name, min_dist, i):
    """The function draw the bounding boxes around the face
        
    Function will show if opencv failed to grab picture
    Parameters
    ----------
    boxes - bounding boxes of all object
    frame - preprocessed frame
    name - name of the recognized subject
    min_dist - minimal gaussian distance between extracted embedding and the embedding 
    inside the database
    i - index of box, which we draw now
    ----------
    
    Returns
    ----------
    True if failed
    False if not
    ----------
    """
    box = boxes[i]
    print(min_dist)
    original_frame = frame[1].copy()  # storing copy of frame before drawing on it
    if min_dist < 0.85:
        data = [str(name), datetime.now().strftime("%H:%M:%S"), min_dist]
        write_to_csv('logger.csv', data)
        print(name)
        frame = cv2.putText(frame[1], str(name) + ' ' + str(min_dist), (int(box[0]), int(box[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (63, 0, 252), 1, cv2.LINE_AA)
        frame = cv2.rectangle(frame[1], (box[0].astype(int), box[1].astype(int)), (box[2].astype(int), box[3].astype(int)),
                              (13, 214, 53), 2)
    else:
        save_face(frame);
    return frame


# In[12]:


def special_cases(k, frame):
    """The function perform keyboard input for special cases 
        
    If user will tap esc, it will close the application
    if space it will save current frame to the folder data2
    
    Parameters
    ----------
    k - input from keyword
    ----------
      
    Returns
    ----------
    True if failed
    False if not
    ----------
    """
    if keyboard.is_pressed('Esc'):
        #k % 256 == 27  # ESC  
        print('Esc pressed, closing...')
        return 0

    elif keyboard.is_pressed('Space'):
        # k % 256 == 32  # space to save image
        print('Enter your name :')
        name = input()

        # create directory if not exists
        if not os.path.exists('imagepath' + name):
            os.mkdir('imagepath' + name)

        img_name = "imagepath/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, frame[1])
        print(" saved: {}".format(img_name))
        return 1
    return 2
