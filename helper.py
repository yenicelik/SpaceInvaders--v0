""" Includes helper functions that are used to pre-process observations, or
    to save figures, lists etc. """

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.color import rgb2gray

import tensorflow as tf

def setup_saver(W, b):
    features = W.copy()
    features.update(b)
    saver = tf.train.Saver(features)

    return saver


#Helper functions
def save_figs(rewardList, stepsList, parameter):
    """ Creates matplotlib figures out of the accumulated lists """
    if not parameter['X11']:
        matplotlib.use('Agg') #hopw this is gonna work

    plt.interactive(False)
    plt.plot(rewardList)
    if parameter['SAVE_FIGS']:
        plt.savefig( 'rewards{:d}.png'.format(i) )
    else:
        plt.ylabel('rewards')
        plt.show()
    
    plt.plot(stepsList)
    if parameter['SAVE_FIGS']:
        plt.savefig( 'steps{:d}.png'.format(i) )
    else:
        plt.ylabel('steps')
        plt.show()

def preprocess_image(observation):  #takes about 20% of the running time!!!
    """ Grayscale, downscale and crop image for less data wrangling """
    #consider transfering this to TF
    out = rgb2gray(observation)    #takes about 5% of the running time!!!               #2s
    out = resize(out, (110, 84))    #takes about 9% of running time!!!
    return out[13:110 - 13, :] 



