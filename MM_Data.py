#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import os
from tqdm import tqdm
import shutil

low = 10000


# In[2]:


from glob import glob
x = glob(r"G:/datasets/Moments_in_Time_Raw/training/*/", recursive = True)


# In[ ]:





# In[3]:


import cv2
def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
        
        
        
def get_frames(title,mypath):
    
    #print(title)
    
    vidcap = cv2.VideoCapture(title)
    
    success,image = vidcap.read()
    
    #print(success)
    count = 0
    while success:
   
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        try:
            cv2.imwrite( mypath + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        except Exception:
            pass
        count = count + 1
    #print("HEYEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",count)
        
    return count


# In[ ]:


for i in tqdm(x):
    print(i)
    y = glob(i+'/*')
    #print(y)
    for j in tqdm(y):
        print("runs")
        os.makedirs( j.split('.')[0])
        c = get_frames(j,j.split('.')[0])
        if(c<low):
            low = c
        remove(j)
        

