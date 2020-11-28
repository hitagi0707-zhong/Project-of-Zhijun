

dataset_path='test_dataset/'
model_path='snowModel_933.h5'
batch_size=1


# Step 1: define the accuracy function

def sAcc(y_true, y_pred):
    min=y_true-0.2
    max=y_true+0.2
    a=K.greater_equal(y_pred,min)
    b=K.less_equal(y_pred,max)
    c=K.equal(a,b)
    return K.mean(c)


# Step 2: load the trained model

from keras.models import load_model
from keras import backend as K

model = load_model(model_path, custom_objects={'sAcc': sAcc} )


# Step 3: load the test dataset

from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = data_gen.flow_from_directory(dataset_path,
                                              target_size = (256,192),
                                              shuffle=False,
                                              class_mode=None,
                                              batch_size=batch_size)


# Step 4: estimate the snow converage of each image

image_path = test_generator.filenames
steps = len(image_path)

pred = model.predict_generator(test_generator, steps=steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
p=pred[:,0]


# Step 5: output the estimation result

import numpy as np
import pandas as pd

answer=np.around(p,decimals=4)

dataframe = pd.DataFrame({'image_path':image_path,'predict_ratio':answer})
dataframe.to_csv("output.csv",index=False,sep=',')

