import numpy as np
import ast
import scipy   
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
def get_ResNet():
    # define ResNet50 model
    model = ResNet50(weights='imagenet')
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract wanted output
    ResNet_model = Model(inputs=model.input, 
        outputs=(model.layers[-4].output, model.layers[-1].output)) 
    return ResNet_model, all_amp_layer_weights

def process(img):
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)
    
def ResNet_CAM(img, model, all_amp_layer_weights):
    # get filtered images from convolutional output + model prediction vector
    last_conv_output, pred_vec = model.predict(process(np.copy(img)))
    # change dimensions of last convolutional outpu tto 7 x 7 x 2048
    last_conv_output = np.squeeze(last_conv_output) 
    # get model's prediction (number between 0 and 999, inclusive)
    pred = np.argmax(pred_vec)
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    # get AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), amp_layer_weights).reshape(224,224) # dim: 224 x 224
    # return class activation map
    return final_output, pred
    
def get_bounds(out, percentile=60):
    # Get bounding box of 95+ percentile pixels
    a = out.flatten()
    filtered = np.array([1 if x > np.percentile(a, percentile) else 0 for x in a]).reshape(224,224)
    left, up, down, right = 224, 224, 0, 0
    for x in range(224):
        for y in range(224):
            if filtered[y,x] == 1:
                left = min(left, x)
                right = max(right, x)
                up = min(up, y)
                down = max(down, y)
    return left, up, down, right

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator()
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(224,224),
        batch_size=1,
        class_mode='categorical')

ResNet_model, all_amp_layer_weights = get_ResNet()

def show_next(heatmap=True, bounds=True):
    fig, ax = plt.subplots()
    global img
    img = test_set.next()[0][0]
    ax.imshow(img, alpha=(0.7 if heatmap else 1.))
    CAM, pred = ResNet_CAM(img, ResNet_model, all_amp_layer_weights)
    if heatmap:
        ax.imshow(CAM, cmap='jet', alpha=0.3)
    if bounds:
        left, up, down, right = get_bounds(CAM)
        import matplotlib.patches as patches
        rect = patches.Rectangle((left, up), (right-left), (down-up), linewidth=1,  edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())
        ax.set_title(imagenet_classes_dict[pred])
    plt.show()
    