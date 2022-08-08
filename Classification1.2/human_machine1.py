
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
import imageio, os, cv2
from tensorflow.keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')

classifier1 = load_model('resources/MachineHuman_model_bak.h5')

def hum_mac(img, Cimg, filename):
	Cimg = Cimg[...,::-1]
	prediction1 = classifier1.predict(img, batch_size=None, steps=1) #gives all class prob.

	if(prediction1[:,:]>0.5):
		imageio.imwrite("result/machine/" + filename, Cimg)
		print(filename + "=======>>>", 'machine')
	else:
		imageio.imwrite("result/human/" + filename, Cimg)
		print(filename + "=======>>>", 'human')



 


		



















