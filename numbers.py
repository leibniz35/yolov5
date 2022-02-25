__author__ = "Mehmet Değirmenci"


import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *





url = ("http://dl.dropboxusercontent.com/s/f1gk6iuvmbxfb1c/numbers.pt?raw=1")
filename = "numbers.pt"
urlretrieve(url,filename)




urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('# Digital Number Detection')
st.write('Currently only works when a single number is loaded')




uploaded_file = st.file_uploader("Upload Files",type=['jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename)
    model.conf = 0.4
    model.conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    
    model.results = model(img_array, size=512)
    model.results.save("yolov5/results")
    model.results.save()
    st.image("yolov5/results/image0.jpg")
    
    
st.write('# Metrics of the model')


url3 = ("http://dl.dropboxusercontent.com/s/pnd6p3zbibx6ofa/map.png?raw=1")
filenameee = "map.png"
urlretrieve(url3,filenameee)
if st.button('mAP'):
    st.image(filenameee)
    
    

url4 = ("http://dl.dropboxusercontent.com/s/0sveaoty719d2ie/losses.png?raw=1")
filenameeee = "losses.png"
urlretrieve(url4,filenameeee)
if st.button('Losses'):
    st.image(filenameeee)


    
    
    
    
    
