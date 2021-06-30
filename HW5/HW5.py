#!/usr/bin/env python
# coding: utf-8

# In[4]:

import random
import os
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import keras
import keras.utils
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout, Input, BatchNormalization, add, GlobalAvgPool2D
from keras.callbacks import ModelCheckpoint, History
from sklearn.metrics import accuracy_score
from keras import optimizers, losses
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
import re
#---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--GPU",type=int, default=0)
parser.add_argument("--epoch",type=int, default=1000)
parser.add_argument("--bs",type=int, default=32)
parser.add_argument("--model_name",type=str, default="ResNet50")
args = parser.parse_args()
#---------------------------------------------------------------------
seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
history = History() 

#-----------------------Make dir----------------------------------------------
dirpath = "./Saved_Model/%s"%(args.model_name)
# if os.path.isdir(dirpath):
#     for file in os.listdir("./Saved_Model/%s"%(args.model_name)):
#         filename = "./Saved_Model/%s/%s"%(args.model_name,file)
#         os.remove(filename)
os.makedirs("Saved_Model/"+args.model_name, exist_ok=True)

#---------------------------------------------------------------------
# ## Load data

# In[5]:

Dataset_root = "./Dataset/"

x_train = np.load(f"{Dataset_root}x_train.npy")
y_train = np.load(f"{Dataset_root}y_train.npy")

x_test = np.load(f"{Dataset_root}x_test.npy")
y_test = np.load(f"{Dataset_root}y_test.npy")

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[6]:


# It's a multi-class classification problem 
class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
               'dog': 5, 'frog': 6,'horse': 7,'ship': 8, 'truck': 9}
print(np.unique(y_train))


# ![image](https://img-blog.csdnimg.cn/20190623084800880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lqcDE5ODcxMDEz,size_16,color_FFFFFF,t_70)

# ## Data preprocess

# In[7]:


x_train = x_train.astype('float32')
print(x_train.shape)

x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to one-hot encoding (keras model requires one-hot label as inputs)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## Build model & training (Keras)

# In[9]:


import warnings
warnings.filterwarnings('ignore')


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size=kernel_size, strides=strides, padding=padding,  kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[2]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    identity = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3], kernel_size=(1,1), strides=(stride_width, stride_height), padding="valid",  kernel_regularizer=regularizers.l2(0.0001))(input)
    return add([identity, residual])

def basic_block(nb_filter, strides=(1, 1)):
    def f(input):
        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3,3))
        return shortcut(input, residual)
    return f

def residual_block(nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2,2)
            input = basic_block(nb_filter, strides)(input)
        return input
    return f

def resnet_18(input_shape=(32, 32, 3), nclass=10):
    input_ = Input(shape=input_shape)
    temp = UpSampling2D(size=(7, 7), interpolation="bilinear")(input_)
    conv1 = conv2d_bn(temp, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
    conv2 = residual_block(64, 2, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 2, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 2, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 2, is_first_layer=True)(conv4)
    pool2 = GlobalAvgPool2D()(conv5)
    output_ = Dense(nclass, activation='softmax')(pool2)
    model = Model(inputs=input_, outputs=output_)
    model.summary()
    return model

def resnet_34(input_shape=(32, 32, 3), nclass=10):
    input_ = Input(shape=input_shape)
    temp = UpSampling2D(size=(7, 7), interpolation="bilinear")(input_)
    conv1 = conv2d_bn(temp, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
    conv2 = residual_block(64, 3, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 4, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 6, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 3, is_first_layer=True)(conv4)
    pool2 = GlobalAvgPool2D()(conv5)
    output_ = Dense(nclass, activation='softmax')(pool2)
    model = Model(inputs=input_, outputs=output_)
    model.summary()
    return model

def resnet_50(input_shape=(32, 32, 3), nclass=10):
    model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = model.layers[-1].output
    x = GlobalAvgPool2D()(x)
    x = Dense(nclass, activation='softmax')(x)
    model = Model(model.input, outputs=x)
    return model

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name="upsample", position='before'):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)
    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})
    # Iterate over all layers after the input
    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]
        # Insert layer if name matches the regular expression
        
        if re.match(layer_regex, layer.name):
            x = model.get_layer("input_1").output
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')
            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer._name = insert_layer_name
            else:
                new_layer._name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)
        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})
    return Model(inputs=model.inputs, outputs=x)

def UpSampling():
    return UpSampling2D(size=(7, 7), interpolation="bilinear")

def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

from keras.preprocessing.image import ImageDataGenerator # inport api
shift = 0.2
# datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift, horizontal_flip=True, vertical_flip=True, rotation_range=30)

# datagen.fit(x_train)
# x_train = datagen.flow(x_train, batch_size=len(x_train))
# x_train = x_train.next()


model = resnet_50()
model = insert_layer_nonseq(model, '.*conv1_pad.*', UpSampling)

plot_model(model, show_shapes=True, show_layer_names=True,to_file='model.png')
save_dir=f"./Saved_Model/{args.model_name}"
filepath = "model.h5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_accuracy', verbose=0, save_best_only=True,mode='auto')
Earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto',patience=15)
Learningrate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-6)
callbacks_list = [checkpoint,history,Learningrate,Earlystop]
# initiate SGD optimizer
opt = keras.optimizers.SGD(learning_rate=1e-2, decay=1e-6, momentum=0.9)
lr_metric = get_lr_metric(opt)
# Compile the model with loss function and optimizer
# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy', lr_metric])




# Fit the data into model
# model.fit(datagen.flow(x_train, y_train, batch_size=args.bs),batch_size=args.bs, epochs=args.epoch, validation_data=(x_test, y_test), callbacks = callbacks_list, verbose = 1, shuffle=True)
model.fit(x_train, y_train, batch_size=args.bs, epochs=args.epoch, validation_data=(x_test, y_test), callbacks = callbacks_list, verbose = 1, shuffle=True)


# In[10]:
Model_save = f"./Saved_Model/{args.model_name}/model.h5"
# loadmodel = load_model(Model_save)
loadmodel = load_model(Model_save, custom_objects={"lr": lr_metric})
y_pred = loadmodel.predict(x_test)
print(y_pred.shape)


# In[11]:


y_pred[0]


# In[12]:


np.argmax(y_pred[0])


# In[13]:


y_pred = np.argmax(y_pred, axis=1)


# ## DO NOT MODIFY CODE BELOW!
# please screen shot your results and post it on your report

# In[14]:


assert y_pred.shape == (10000,)


# In[15]:


y_test = np.load(f"{Dataset_root}y_test.npy")
print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))


# In[ ]:




