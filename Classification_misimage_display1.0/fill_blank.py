
import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
import imageio, cv2
from tensorflow.keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')

classifier1 = load_model('resources/blankfill_model_bak.h5')

def belong_word(text, word):
	if word in text:
		return True
	else:
		return False

def fill_blank(img1, Cimg, name_extension):
	Cimg1 = Cimg[...,::-1]
	prediction1 = classifier1.predict(img1, batch_size=None, steps=1) #gives all class prob.
	
	if(prediction1[:,:]>0.5): #fill
		
		if belong_word(name_extension, 'blank') :
			imageio.imwrite("result/mis_fill/" + name_extension, Cimg1)
			return "blank"
		else :
			return "fill"
	else: #blank
		
		if belong_word(name_extension, 'blank') :
			imageio.imwrite("result/blank/" + name_extension, Cimg1)
			return "blank"
		else :
			imageio.imwrite("result/mis_blank/" + name_extension, Cimg1)
			return "fill"




 


		



















