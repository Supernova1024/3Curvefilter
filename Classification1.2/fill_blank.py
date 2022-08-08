
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
import imageio, cv2
from tensorflow.keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')

classifier1 = load_model('resources/blankfill_model_bak.h5')

def fill_blank(img1, Cimg, name_extension):
    Cimg = Cimg[...,::-1]
    prediction1 = classifier1.predict(img1, batch_size=None, steps=1) #gives all class prob.
    
    if(prediction1[:,:]>0.5):
        print(name_extension + "=======>>>", 'fill')
        return "fill"
    else:
        imageio.imwrite("result/blank/" + name_extension, Cimg)
        print(name_extension + "=======>>>", 'blank')
        return "blank"




 


        



















