# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:34:04 2018
"""

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('./Keras_Model.h5')
test_image = image.load_img('test_0.jpg', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'sunflower'
else:
    prediction ='rose'
print(prediction)
