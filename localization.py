# Localization

from keras.models import load_model, Model

layer_index = -4

classifier = load_model('model.h5')
weights = classifier.layers[-1].get_weights()[0]
classifier = Model(inputs=classifier.input,outputs=(classifier.layers[layer_index].output,classifier.layers[-1].output))

if layer_index >= 0:
    layer_index += 1
width_factor = classifier.layers[layer_index].output_shape[1]
height_factor = classifier.layers[layer_index].output_shape[2]

# Load image
import numpy as np

import matplotlib.pyplot as plt
def visualize(img):
    conv_out, guess = classifier.predict(img)
    print(guess)
    conv_out = np.squeeze(conv_out)
    for i in range(len(conv_out[0][0])):
        plt.figure()
        plt.imshow(conv_out[:,:,i])


def get_heatmap(image):
    conv_out, pred = classifier.predict(image)
    conv_out = np.squeeze(conv_out)
    pred = np.argmax(pred)
    from scipy.ndimage import zoom
    mat_for_mult = zoom(conv_out, (64./width_factor, 64./height_factor, 1), order=1)
    weights = classifier.layers[-1].get_weights()[0][:,int(pred)]
    out = np.dot(mat_for_mult.reshape((64*64, 32)), weights).reshape((64, 64))
    return pred, out

def get_bounds(out, percentile=95):
    # Get bounding box of 95+ percentile pixels
    a = out.flatten()
    filtered = np.array([1 if x > np.percentile(a, percentile) else 0 for x in a]).reshape(64,64)
    left, up, down, right = 64, 64, 0, 0
    for x in range(64):
        for y in range(64):
            if filtered[y,x] == 1:
                left = min(left, x)
                right = max(right, x)
                up = min(up, y)
                down = max(down, y)
    return left, up, down, right

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical')

def load_image(path):
    from PIL import Image
    img = Image.open(path)
    img = img.resize((64,64))
    arr = np.asarray(img)
    arr = arr/255.0
    return arr

def show_next(bounds=True, heatmap=True, show_image=True, image=[]):
    if len(image) == 0:
        img = test_set.next()[0]
    else:
        img = image
        img = np.expand_dims(img, axis=0)
    global out
    pred, out = get_heatmap(img)  
    
    # Plot Heatmap
    fig, ax = plt.subplots()
    ax.imshow(img[0], alpha=(0.7 if heatmap else 1.))
    if heatmap:
        ax.imshow(out, cmap='jet', alpha=(0.3 if show_image else 1.))
    if bounds:
        left, up, down, right = get_bounds(out)
        import matplotlib.patches as patches
        rect = patches.Rectangle((left, up), (right-left), (down-up), linewidth=1,  edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if pred == 0:
        ax.set_title('Cat')
    else:
        ax.set_title('Dog')
    plt.show()
    