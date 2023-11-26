#%% Import packages
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#%% Step 5: Model Testing

script_path = os.path.realpath(__file__)
root = os.path.dirname(script_path)
test1_path = os.path.join(root,'Data','Test','Medium',
                          'Crack__20180419_06_19_09,915.bmp')
test2_path = os.path.join(root,'Data','Test','Large',
                          'Crack__20180419_13_29_14,846.bmp')

test1 = image.img_to_array(image.load_img(test1_path, target_size=(100, 100)))
test2 = image.img_to_array(image.load_img(test2_path, target_size=(100, 100)))
test1 = np.expand_dims(test1/255.0, axis=0)
test2 = np.expand_dims(test2/255.0, axis=0)

model = tf.keras.models.load_model('saved_model_Duong_65857.h5')

pred1_prob = model.predict(test1)
pred1 = np.argmax(pred1_prob, axis=-1)

pred2_prob = model.predict(test2)
pred2 = np.argmax(pred2_prob, axis=-1)

classes = ['Large','Medium','Small','None']

text_position = (10, 100)
text_color = 255
font_path = os.path.join(root,'DejaVuSansMono.ttf')
font = ImageFont.truetype(font_path, 76)

img1 = Image.open(test1_path)
draw = ImageDraw.Draw(img1)
text1 = 'True Crack Classification Label: '+classes[pred1[0]]+'\n'\
        +'Predicted Crack Classification Label: Medium\n\n\n\n\n\n\n\n\n\n\n\n'\
        +'\n\n\n\n\n\n\nLarge Crack: '+str(round(pred1_prob[0,0]*100,2))+'%\n'\
        +'Medium Crack: '+str(round(pred1_prob[0,1]*100,2))+'%\n'\
        +'Small Crack: '+str(round(pred1_prob[0,2]*100,2))+'%\n'\
        +'No Crack: '+str(round(pred1_prob[0,3]*100,2))+'%\n'
draw.text(text_position, text1, fill=text_color, font=font)
img1.show()
img1.save(os.path.join(root,'Test Prediction 1:Crack__20180419_06_19_09,915.bmp'))


img2 = Image.open(test2_path)
draw = ImageDraw.Draw(img2)
text2 = 'True Crack Classification Label: '+classes[pred2[0]]+'\n'\
        +'Predicted Crack Classification Label: Large\n\n\n\n\n\n\n\n\n\n\n\n'\
        +'\n\n\n\n\n\n\nLarge Crack: '+str(round(pred2_prob[0,0]*100,2))+'%\n'\
        +'Medium Crack: '+str(round(pred2_prob[0,1]*100,2))+'%\n'\
        +'Small Crack: '+str(round(pred2_prob[0,2]*100,2))+'%\n'\
        +'No Crack: '+str(round(pred2_prob[0,3]*100,2))+'%\n'
draw.text(text_position, text2, fill=text_color, font=font)
img2.show()
img2.save(os.path.join(root,'Test Prediction 2: Crack__20180419_13_29_14,846.bmp'))

