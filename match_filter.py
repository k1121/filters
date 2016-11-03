# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 15:07:36 2016

@author: temp
"""
import os
from scipy.misc import imread
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from scipy import signal
import time

imagesPath = r'D:\Projects\python\match_filter'

class filters():
    
    def form_train_set(self,directory_with_images, cash = True):
       images_paths = [os.path.join(directory_with_images,image_name) for image_name in os.listdir(directory_with_images) if os.path.splitext(image_name)[1] in ['.jpg','png','tif']]
       if cash == True:
            for k, image_path in enumerate(images_paths):
                if k == 0:                    
                    image = myfft(imread(image_path))
                    self.training_image_shape = image.shape
                    images_matrix = [np.reshape(image, self.training_image_shape[0]*self.training_image_shape[1],1)]
                    continue
                image = myfft(imread(image_path))
                images_matrix.append(np.reshape(image, self.training_image_shape[0]*self.training_image_shape[1],1))
            return images_matrix
    def form_true_class_set(self, directory_with_images, cash = True):
        self.true_class_set = self.form_train_set(directory_with_images, cash)
        
    def form_false_class_set(self, directory_with_images, cash = True):
        self.true_class_set = self.form_train_set(directory_with_images, cash)
            
    def predict(self,input_images):
        pass
tr = filters()
tr.form_true_class_set(r'D:\Projects\python\match_filter\true_class')

class match_filter(filters):
    pass
    
        

def mesh(matrix):
    ax = axes3d.Axes3D(plt.figure())    
    x = np.arange(0,len(matrix))
    X,Y = np.meshgrid(x,x)
    ax.plot_surface(X, Y, matrix, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)   


def fcor(image1, image2):
    image1_shape = image1.shape
    image2_shape = image2.shape
    image1_big = np.pad(image1,(image2_shape[0]/2,image2_shape[1]/2),'constant')
    image2_big = np.pad(image2,(image1_shape[0]/2,image1_shape[1]/2),'constant')
    fourie_image1 = fft2(image1_big)
    fourie_image2 = fft2(image2_big)
    cor = fftshift(ifft2(fourie_image1*np.conj(fourie_image2)))
    return cor
    
    
#start_time = time.time()
#c1 = fcor(images_matrix[0], images_matrix[1])
#print time.time() - start_time
#
#start_time = time.time()
#c2 = signal.correlate2d(np.complex128(images_matrix[0]), np.complex128(images_matrix[1]))
#print time.time() - start_time
##imshow(np.abs(c1), interpolation='none')
#mesh(np.abs(c1))
##imshow(np.abs(c2), interpolation='none')
#mesh(np.abs(c2))
#plt.show()