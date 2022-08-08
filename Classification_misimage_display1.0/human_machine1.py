
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
import imageio, os, cv2
from tensorflow.keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')

classifier1 = load_model('resources/MachineHuman_model_bak.h5')

def belong_word(text, word):
	if word in text:
		return True
	else:
		return False

def hum_mac(img, Cimg, filename):
	Cimg = Cimg[...,::-1]
	prediction1 = classifier1.predict(img, batch_size=None, steps=1) #gives all class prob.

	if(prediction1[:,:]>0.5): #machine
		if belong_word(filename, 'machine') :
			imageio.imwrite("result/machine/" + filename, Cimg)
		else :
			imageio.imwrite("result/mis_machine/" + filename, Cimg)

	else: #human
		if belong_word(filename, 'human') :
			imageio.imwrite("result/human/" + filename, Cimg)
		else :
			imageio.imwrite("result/mis_human/" + filename, Cimg)



 


		



















