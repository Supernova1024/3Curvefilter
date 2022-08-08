#!/usr/bin/python
# from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2, os
import math
import numpy as np
from scipy import stats
import imageio
from resizeimage import resizeimage
from PIL import Image
from skimage import filters
import cv2 as cv
import tensorflow
from tensorflow.keras.preprocessing import image
import time
from numpy import array

from human_machine1 import hum_mac
from fill_blank import fill_blank
import threading
import time


def caculate_time_difference(start_milliseconds, end_milliseconds, filename):
   if filename == 'total':
      diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
      seconds=(diff_milliseconds / 1000) % 60
      minutes=(diff_milliseconds/(1000*60))%60
      hours=(diff_milliseconds/(1000*60*60))%24
      print("Total run time", hours,":",minutes,":",seconds)
   else:
      diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
      seconds=(diff_milliseconds / 1000) % 60
      print(seconds, "s", filename)

def input_prepare(img):
   img = cv2.resize(img, (480, 640))
   img = img[...,::-1]
   img = img / 255
   img = np.expand_dims(img, axis=0) 
   return img 

def crop_img(img):
   h, w = img.shape[:2]
   cropped_img = img[int(h/2) : h, 0 : int(w/2)]
   return cropped_img

global thread_kill_flags

def processing(threadID, files, d):
   for filename in files:
      start_milliseconds = str(int(round(time.time() * 1000)))
      Cimg = cv2.imread("image/" + filename)
      if Cimg is not None:
         if d== 'quad':
            Cimg = crop_img(Cimg)
            Cimg = Cimg[...,::-1]
         input_img = input_prepare(Cimg)
         fill = fill_blank(input_img, Cimg, filename)
         if fill == 'fill':
            hum_mac(input_img, Cimg, filename)
      end_milliseconds = str(int(round(time.time() * 1000)))
      caculate_time_difference(start_milliseconds, end_milliseconds, filename)

class myThread (threading.Thread):
   def __init__(self, threadID, name, files, d, start_time):
      threading.Thread.__init__(self)
      self._stop = threading.Event()
      self.threadID = threadID
      self.name = name
      self.files = files
      self.d = d
      self.start_time = start_time
   def stop(self):
      self._stop.set()

   def run(self):
      print("====================", self.name)
      processing(self.threadID, self.files, self.d)
      self.stop()
      end = str(int(round(time.time() * 1000)))
      caculate_time_difference(self.start_time, end, 'total')

def main(folder, d, start_time):
   stop_threads = False 
   filenames = []
   thread_list = []
   count_thread = 10
   
   for filename in os.listdir(folder):
      filenames.append(filename)

   mode = len(filenames) % (count_thread - 1)
   step = len(filenames) / (count_thread - 1)

   if len(filenames) < 20:
      print("Image classifying.....")
      start_total = str(int(round(time.time() * 1000)))
      for filename in os.listdir(folder):
         start_milliseconds = str(int(round(time.time() * 1000)))
         Cimg = cv2.imread(os.path.join(folder, filename))
         if Cimg is not None:
            if d== 'quad':
               Cimg = crop_img(Cimg)
               Cimg = Cimg[...,::-1]
            input_img = input_prepare(Cimg)
            fill = fill_blank(input_img, Cimg, filename)
            if fill == 'fill':
               hum_mac(input_img, Cimg, filename)
         end_milliseconds = str(int(round(time.time() * 1000)))
         caculate_time_difference(start_milliseconds, end_milliseconds, filename)
      end_total = str(int(round(time.time() * 1000)))
      caculate_time_difference(start_total, end_total, "total")
   else:
      for i in range(1, count_thread):
         files = filenames[int(step)*(i-1) : int(step)*i]
         # Create new threads
         thread = myThread(i, "Thread_"+str(i), files, d, start_time)
         thread_list.append(thread)

      # Start new Threads
      for thread in thread_list:
         thread.start()

      if mode != 0:
         # Start mode Threads
         files = filenames[(count_thread - 1)*int(step):]
         thread1 = myThread(count_thread, "Thread_"+str(count_thread), files, d, start_time)
         thread1.start()

   
if __name__ == '__main__':
   start_time = str(int(round(time.time() * 1000)))
   # construct the argument parse and parse the arguments
   ap = argparse.ArgumentParser()
   ap.add_argument("-d", "--debug", required=False,
      help="path to the input image")
   args = vars(ap.parse_args())
   debug = args["debug"]
   print(debug)
   if debug != 'isolated' and debug != 'quad':
      print("ERROR:: Please enter the type of image correctly!")
   else :
      main("image", debug, start_time)
