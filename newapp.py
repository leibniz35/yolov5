import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy
import matplotlib.pyplot as plt
import torch
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *
import cv2
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st


url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)



urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('# LÖSEMİ TESPİT PLATFORMU ')


uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename)

   
    
    model.results = model(img_array, size=640)
    model.results.save()
    st.image("runs/hub/exp24/image0.jpg")
  
    liste = []
    liste1 = []
    liste2 = []
    liste0 = []


    for i in model.results.xywh:
        for j in i:
            for k in j:
                liste.append(k)
            if k ==2:
                liste2.append(k)
            elif k == 1:
                liste1.append(k)
            elif k == 0:
                liste0.append(k)


    st.write("Saptanan hücre sayısı:")
    st.write("Beyaz Kan Hücresi:",len(liste0))
    st.write("Kırmızı Kan hücresi",len(liste1))
    st.write("Blast hücresi",len(liste2))
