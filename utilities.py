import keras
import os
import random
import pandas as pd
import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization, Dense,Dropout,Flatten
from keras import models
from keras import regularizers
from keras import optimizers
from keras.preprocessing import image

def get_files(directory):
    '''
    Gets the number of images in a directory.
    input: a directory to get files from
    output: returns the number of images in the directory
    '''
    if not os.path.exists(directory):
        return 0
    count=0
    for current_path,dirs,files in os.walk(directory):
        for dr in dirs:
            count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
    return count

def get_dictionary(directory):
    '''
    A CSV is saved with the file name and number of files.
    It is saved in a dictionary, sorted, converted to a df and saved as a csv.
    input: the directory
    output: a saved csv files
    '''
    def get_num(directory):
        name_list = []
        num_files = []
        dictionary = {}
        for file in glob.glob(directory+'/*'):
            file_name = file.split('/')[2]
            name_list.append(file_name)
            count = len(glob.glob(file+'/*'))
            num_files.append(count)
        for k,v in zip(name_list, num_files):
            dictionary[k] = v
        return dictionary
    dictionary = get_num(directory)
    sorted_dict = {k:v for k,v in sorted(dictionary.items(), key = lambda item: item[1])}
    df = pd.DataFrame(list(sorted_dict.items()),columns = ['Plant','Num_images'])
    df.to_csv(str(directory)+'df.csv')

def get_dataframe(file_list, directory):
    '''
    Get a csv file containing the file name, disease id, and disease type.
    input: a list of labels, and the directory of interest
    output: a saved csv file with filename, disease id, and disease type
    '''
    data_list = []
    for defects_id, sp in enumerate(file_list):
        for file in os.listdir(os.path.join(directory, sp)):
            data_list.append(['{}/{}'.format(sp,file), defects_id,sp])
    return pd.DataFrame(data_list, columns = ['File','Disease_id', 'Disease_Type'])

def read_img(filepath, size):
    '''
    Loads an image when given a filepath. Returns the image as an array.
    input: filepath and the target size of the image
    output: returns the image as an array
    '''
    train_dir = 'data_withaug/output/train'
    val_dir = 'data_withaug/output/val'
    test_dir = 'data_withaug/output/test'
    img = image.load_img(train_dir+'/'+filepath, target_size = size)
    img = image.img_to_array(img)
    return img
