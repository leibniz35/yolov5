import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *


url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)

urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('# KAN HÜCRESİ TESPİTİ')


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
    model.results.save()  # or .show()
    st.image("runs/hub/exp")
  
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


    st.write("The number of detected WBC is",len(liste2),"The number of detected RBC is",len(liste1),"The number of detected PLT is",len(liste0))
