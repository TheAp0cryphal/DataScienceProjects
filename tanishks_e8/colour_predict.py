#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.color import rgb2lab
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys


OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_rgb:.3f} {bayes_convert:.3f}\n'
    'kNN classifier:         {knn_rgb:.3f} {knn_convert:.3f}\n'
    'Rand forest classifier: {rf_rgb:.3f} {rf_convert:.3f}\n'
)


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
    y_grid = model.predict(X_grid.reshape((-1, 3)))
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
    plt.imshow(X_grid.reshape((hei, wid, -1)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def main():
    data = pd.read_csv(sys.argv[1])
    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values

    
    def toLab(val):
        val = val.reshape(1, -1, 3)
        val = rgb2lab(val)
        val = val.reshape(-1, 3)
        return val
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)    
    
    model_lab_bayes = make_pipeline(
        FunctionTransformer(toLab, validate = True),
        GaussianNB()
    )
    
    model_rgb_bayes = GaussianNB()    
    
    
    model_lab_knn = make_pipeline(
        FunctionTransformer(toLab, validate = True),
        KNeighborsClassifier (n_neighbors = 20)
    )
    
    model_rgb_knn = KNeighborsClassifier (n_neighbors = 20)
    
    model_lab_rfc = make_pipeline(
        FunctionTransformer(toLab, validate = True),
        RandomForestClassifier (n_estimators = 500, max_depth=10)
    )
    
    model_rgb_rfc = RandomForestClassifier(n_estimators = 500, max_depth=10)
    
    

    # train each model and output image of predictions
    models = [model_rgb_bayes, model_lab_bayes, model_rgb_knn, model_lab_knn, model_rgb_rfc, model_lab_rfc]
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        plot_predictions(m)
        plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=model_rgb_bayes.score(X_valid, y_valid),
        bayes_convert=model_lab_bayes.score(X_valid, y_valid),
        knn_rgb=model_rgb_knn.score(X_valid, y_valid),
        knn_convert=model_lab_knn.score(X_valid, y_valid),
        rf_rgb=model_rgb_rfc.score(X_valid, y_valid),
        rf_convert=model_lab_rfc.score(X_valid, y_valid),
    ))


if __name__ == '__main__':
    main()


# In[ ]:




