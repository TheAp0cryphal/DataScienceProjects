#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from skimage.color import rgb2lab
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 113, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 186, 186),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def main(infile):
    data = pd.read_csv(infile)
    X = data[['R', 'G', 'B']].to_numpy() / 255 # array with shape (n, 3). Divide by 255 so components are all 0-1.
    y = data['Label'].to_numpy() # array with shape (n,) of colour words.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    # TODO: build model_rgb to predict y from X.
    model_rgb = GaussianNB()
    model_rgb.fit(X_train, y_train)
    predicted = model_rgb.predict(X_valid)
    print("First Accuracy: ")
    print(accuracy_score(predicted, y_valid), "\n")    

    def toLab(val):
        val = val.reshape(1, -1, 3)
        val = rgb2lab(val)
        val = val.reshape(-1, 3)
        return val
    
    # TODO: build model_lab to predict y from X by converting to LAB colour first.
    model_lab = make_pipeline(
        FunctionTransformer(toLab, validate = True),
        GaussianNB()
    )
    model_lab.fit(X_train, y_train)
    predicted1 = model_lab.predict(X_valid)
    print("Second Accuracy:")
    print(accuracy_score(predicted1, y_valid))
    # TODO: print model_lab's accuracy score

    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main(sys.argv[1])


# In[ ]:





# In[ ]:




