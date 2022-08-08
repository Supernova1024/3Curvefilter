#!/usr/bin/env python
# coding: utf-8

# # Machine vs Human Convolution Neural Network Classifier
# 
# ### Problem statement :
# 
# In this Section we are implementing Convolution Neural Network(CNN) Classifier for Classifying Machine and Human images. The Total number of images available for training is 25,000 and final testing is done on seperate 10,000 images.
# #### Note:This problem statement and dataset is taken from [this](https://www.kaggle.com/c/Machines-vs-Humans) Kaggle competition.
# 
# ### Dependencies
# * Jupyter notebook
# * Tensorflow 1.10
# * Python 3.6
# * Matplotlib
# * Seaborn
# * Scikit-Learn
# * Pandas
# * Numpy
# 
# Install dependencies using [conda](https://conda.io/docs/)
# 
# #### Test Train Split
# Image training set contain 12500 images for each Humanegory. I split those into 80% train and 20% means test Split each class images into 10,000 for train and 2,500 for test. 
# 
# ### Architecture
# ![image](resources/architecture.jpg)

# In[1]:

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


# In[2]:


from warnings import filterwarnings
filterwarnings('ignore')


# Network Parameter:
# * Rectifier Linear Unit 
# * Adam optimizer
# * Sigmoid on Final output
# * Binary CrossEntropy loss

# In[3]:


# classifier = Sequential()
# classifier.add(Conv2D(32,(3,3),input_shape=(640,480,3),activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
# classifier.add(Conv2D(32,(3,3),activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
# classifier.add(Flatten())
# classifier.add(Dense(units=128,activation='relu'))
# classifier.add(Dense(units=1,activation='sigmoid'))
# adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# ## Data Augmentation
# Using some Data Augmentation techniques for more data and Better results.
# * Shearing of images
# * Random zoom
# * Horizontal flips

# In[4]:
from tensorflow.keras.models import load_model
classifier = load_model('resources/MachineHuman_model_bak.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set
train_set = train_datagen.flow_from_directory('train',
                                             target_size=(640,480),
                                             batch_size=32,
                                             class_mode='binary')
#Validation Set
test_set = test_datagen.flow_from_directory('validation',
                                           target_size=(640,480),
                                           batch_size = 32,
                                           class_mode='binary',
                                           shuffle=False)
#Test Set /no output available
test_set1 = test_datagen.flow_from_directory('test',
                                            target_size=(640,480),
                                            batch_size=32,
                                            shuffle=False)


# In[5]:


# get_ipython().run_cell_magic('capture', '', "classifier.fit_generator(train_set,\n                        steps_per_epoch=50, \n                        epochs = 20,\n                        validation_data = test_set,\n                        validation_steps = 20, \n                        #callbacks=[tensorboard]\n                        );\n\n#Some Helpful Instructions:\n\n#finetune you network parameter in last by using low learning rate like 0.00001\n#classifier.save('resources/MachineHuman_model_bak.h5')\n#from tensorflow.keras.models import load_model\n#model = load_model('partial_trained1')\n#100 iteration with learning rate 0.001 and after that 0.0001")

# %%capture
classifier.fit_generator(train_set,
                        steps_per_epoch=61, 
                        epochs = 30,
                        validation_data = test_set,
                        validation_steps = 20, 
                        #callbacks=[tensorboard]
                        );

#Some Helpful Instructions:

#finetune you network parameter in last by using low learning rate like 0.00001
classifier.save('resources/MachineHuman_model_bak.h5')
#from tensorflow.keras.models import load_model
# model = load_model('partial_trained1')
#100 iteration with learning rate 0.001 and after that 0.0001
# In[6]:

from tensorflow.keras.models import load_model
classifier = load_model('resources/MachineHuman_model_bak.h5')


# ### Prediction of Single Image

# In[7]:


#Prediction of image
# %matplotlib inline
import tensorflow
import tensorflow
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
img1 = image.load_img('validation/human/human1 (2).jpg', target_size=(640,480))
img = image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
if(prediction[:,:]>0.5):
    value ='Machine :%1.2f'%(prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='Human :%1.2f'%(1.0-prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
plt.imshow(img1)
# plt.show()


# In[8]:


import pandas as pd
test_set.reset
ytesthat = classifier.predict_generator(test_set)
df = pd.DataFrame({
    'filename':test_set.filenames,
    'predict':ytesthat[:,0],
    'y':test_set.classes
})


# In[9]:


pd.set_option('display.float_format', lambda x: '%.5f' % x)
df['y_pred'] = df['predict']>0.5
df.y_pred = df.y_pred.astype(int)
df.head(10)
print(df)

# In[10]:


misclassified = df[df['y']!=df['y_pred']]
print('Total misclassified image from 10 Validation images : %d'%misclassified['y'].count())


# In[11]:


#Prediction of test set
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(df.y,df.y_pred)
sns.heatmap(conf_matrix,cmap="YlGnBu",annot=True,fmt='g');
plt.xlabel('predicted value')
plt.ylabel('true value');


# In[12]:


#Some of Human image misclassified as Machine.
import matplotlib.image as mpimg

HumanasMachine = df['filename'][(df.y==0)&(df.y_pred==1)]
print("====", HumanasMachine)
fig=plt.figure(figsize=(15, 6))
columns = len(HumanasMachine) - 1
rows = 1
for i in range(columns*rows):
    #img = mpimg.imread()
    img = image.load_img('validation/'+HumanasMachine.iloc[i], target_size=(640,480))
    
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)

# plt.show()

# In[13]:


#Some of Machine image misclassified as Human.
import matplotlib.image as mpimg

MachineasHuman = df['filename'][(df.y==1)&(df.y_pred==0)]
fig=plt.figure(figsize=(15, 6))


columns = len(HumanasMachine) - 1
rows = 1
for i in range(columns*rows):
    #img = mpimg.imread()
    img = image.load_img('validation/'+MachineasHuman.iloc[i], target_size=(640,480))
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
# plt.show()


# In[14]:


classifier.summary()

# ### Visualization of Layers Ouptut
# 

# In[15]:


#Input Image for Layer visualization
img1 = image.load_img('validation/human/human1 (2).jpg')
plt.imshow(img1);
#preprocess image
img1 = image.load_img('validation/human/human1 (2).jpg', target_size=(640,480))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)


# In[16]:


model_layers = [ layer.name for layer in classifier.layers]
print('layer name : ',model_layers)


# In[17]:


from tensorflow.keras.models import Model
conv2d_6_output = Model(inputs=classifier.input, outputs=classifier.get_layer('conv2d').output)
conv2d_7_output = Model(inputs=classifier.input,outputs=classifier.get_layer('conv2d_1').output)


# In[18]:


conv2d_6_features = conv2d_6_output.predict(img)
conv2d_7_features = conv2d_7_output.predict(img)
print('First conv layer feature output shape : ',conv2d_6_features.shape)
print('First conv layer feature output shape : ',conv2d_7_features.shape)


# ### Single Convolution Filter Output

# In[19]:


plt.imshow(conv2d_6_features[0, :, :, 4], cmap='gray')


# ### First Covolution Layer Output

# In[20]:


import matplotlib.image as mpimg

fig=plt.figure(figsize=(14,7))
columns = 5
rows = 2
for i in range(columns*rows):
    #img = mpimg.imread()
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_6_features[0, :, :, i], cmap='gray')
# plt.show()


# ### Second Covolution Layer Output

# In[21]:


fig=plt.figure(figsize=(14,7))
columns = 5
rows = 2
for i in range(columns*rows):
    #img = mpimg.imread()
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_7_features[0, :, :, i], cmap='gray')
# plt.show()


# ### Model Performance on Unseen Data

# In[22]:


# for generator image set u can use 
# ypred = classifier.predict_generator(test_set)

fig=plt.figure(figsize=(15, 6))
columns = 3
rows = 3
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img1 = image.load_img('test/'+test_set1.filenames[np.random.choice(range(10))], target_size=(640,480))
    img = image.img_to_array(img1)
    img = img/255
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
    if(prediction[:,:]>0.5):
        value ='Machine :%1.2f'%(prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    else:
        value ='Human :%1.2f'%(1.0-prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))

    plt.imshow(img1)
plt.show()


# In[23]:


# get_ipython().run_cell_magic('capture', '', '# Model Accuracy\nx1 = classifier.evaluate_generator(train_set)\nx2 = classifier.evaluate_generator(test_set)')
# %%capture
# Model Accuracy
x1 = classifier.evaluate_generator(train_set)
x2 = classifier.evaluate_generator(test_set)

# In[24]:


print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f'%(x1[1]*100,x1[0]))
print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f'%(x2[1]*100,x2[0]))


# ### Conclusion
# The Architecture and parameter used in this network are capable of producing accuracy of **97.56%** on Validation Data which is pretty good. It is possible to Achieve more accuracy on this dataset using deeper network and fine tuning of network parameters for training. You can download this trained model from resource directory and Play with it. 
