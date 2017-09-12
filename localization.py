# Localization

from keras.models import load_model, Model

layer_index = -4

classifier = load_model('model_categorical_90.h5')
weights = classifier.layers[-1].get_weights()[0]
classifier = Model(inputs=classifier.input,outputs=(classifier.layers[layer_index].output,classifier.layers[-1].output))

if layer_index >= 0:
    layer_index += 1
width_factor = classifier.layers[layer_index].output_shape[1]
height_factor = classifier.layers[layer_index].output_shape[2]

img_path = 'dataset/single_prediction/cat_or_dog_1.jpg'

# Load image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(img_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

import matplotlib.pyplot as plt
def visualize(classifier, img):
    conv_out, guess = classifier.predict(img)
    print(guess)
    conv_out = np.squeeze(conv_out)
    for i in range(len(conv_out[0][0])):
        plt.figure()
        plt.imshow(conv_out[:,:,i])

# Get heatmap
conv_out, pred = classifier.predict(test_image)
conv_out = np.squeeze(conv_out)
pred = np.argmax(pred)
from scipy.ndimage import zoom
mat_for_mult = zoom(conv_out, (64./width_factor, 64./height_factor, 1), order=1)
weights = classifier.layers[-1].get_weights()[0][:,int(pred)]
out = np.dot(mat_for_mult.reshape((64*64, 32)), weights).reshape((64, 64))

import cv2
im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (64, 64))
"""
# Plot Heatmap
fig, ax = plt.subplots()
# load image, convert BGR --> RGB, resize image to 224 x 224,
# plot image
ax.imshow(im, alpha=0.5)
# get class activation map
CAM = out
# plot class activation map
ax.imshow(CAM, cmap='jet', alpha=0.5)
# obtain the predicted ImageNet category
if pred == 0:
    ax.set_title('Cat')
else:
    ax.set_title('Dog')
"""
    
# Get bounding box of 95+ percentile pixels
a = [max(0,x) for x in out.flatten()]
b = [x for x in a if x > 0]
filtered = np.array([1 if x > np.percentile(b, 95) else 0 for x in a]).reshape(64,64)
left, up, down, right = 64, 64, 0, 0
for x in range(64):
    for y in range(64):
        if filtered[x,y] == 1:
            left = min(left, x)
            right = max(right, x)
            up = min(up, y)
            down = max(down, y)
            
# Plot bounding box on top of image
import matplotlib.patches as patches
fig,ax = plt.subplots()
ax.imshow(im)
rect = patches.Rectangle((left, up), (right-left), (down-up), linewidth=1,  edgecolor='r', facecolor='none')
ax.add_patch(rect)
if pred == 0:
    ax.set_title('Cat')
else:
    ax.set_title('Dog')
plt.show()