import streamlit as st
from PIL import Image
from urllib.request import urlretrieve
from fastai.vision.widgets import *
from fastai.vision.all import *





url = ("http://dl.dropboxusercontent.com/s/w20vwd3bu7fo4ut/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)



urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)
st.write('# LEUKEMIA DETECTION PLATFORM ')


uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=filename)
    model.conf = 0.75
    
    #model.results = model(img_array, size=512)
    #model.results.save("yolov5/results")
    #model.results.save()
    ######
  if st.sidebar.button("rbc"):
        model.classes = 1
        model.results = model(img_array, size=512)
        model.results.save("yolov5/results")
        st.image("yolov5/results/image0.jpg")
        
  
    elif st.sidebar.button("wbc"):
        model.classes = 0
        model.results = model(img_array, size=512)
        model.results.save("yolov5/results")
        st.image("yolov5/results/image0.jpg")
 

    elif st.sidebar.button("blast"):
        model.classes = 2
        model.results = model(img_array, size=512)
        model.results.save("yolov5/results")
        st.image("yolov5/results/image0.jpg")
        
        
        
        
    elif st.sidebar.button("wbc and blast"):
        model.classes = 2,0
        model.results = model(img_array, size=512)
        model.results.save("yolov5/results")
        st.image("yolov5/results/image0.jpg")
        
        
        
    elif st.sidebar.button("All"):
        model.classes = 0,1,2
        model.results = model(img_array, size=512)
        model.results.save("yolov5/results")
        st.image("yolov5/results/image0.jpg")
        
        
        
    model.conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
        


        

        
  

 
       
 
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


    st.write("Number of cells detected:")
    st.write("White Blood Cell:",len(liste0))
    st.write("Red Blood Cell:",len(liste1))
    st.write("Blast Cell",len(liste2))
    
    
    




