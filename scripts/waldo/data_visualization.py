<<<<<<< HEAD
# Copyright      2018  Johns Hopkins University (author: Daniel Povey)

# Apache 2.0


""" TODO
"""
"""randomly coloring white areas. low is 10 as a threshold.
 dimensions are = Red, Blue, Green, Alpha
 Alpha is set 38 for a ~20% transparency"""
def visualize_mask(x):
    validate_image_with_mask(x)
    mask = x['mask']
    red, green, blue, alpha = mask.T
    white_areas = (red > 0) & (blue > 0) & (green > 0)
    black_areas = (red == 0) & (blue == 0) & (green == 0) & (alpha == 255)

    mask[..., :][white_areas.T] = (
        np.random.randint(low=10, high=255), np.random.randint(low=10, high=255),
        np.random.randint(low=10, high=255), 38)
    mask[..., :][black_areas.T] = 0

    x['mask'] = mask
    validate_image_with_mask(x)
    return None

    """This function accepts an object x that should represent an image with a
       mask, and it modifies the image to superimpose the "mask" on it.  The
       image will still be visible through a semi-transparent mask layer.
       This function returns None; it modifies x in-place.
    """
    validate_image_with_mask(x)
    # ... do something, modifying x somehow
    return None
=======
# Copyright      2018  Johns Hopkins University (author: Daniel Povey, Desh Raj, Adel Rahimi)

# Apache 2.0
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

from waldo.data_types import *


def visualize_mask(x, c, transparency=0.3):
    """
    This function accepts an object x that should represent an image with a
    mask, a config class c, and a float 0 < transparency < 1.  
    It changes the image in-place by overlaying the mask with transparency 
    described by the parameter.
    x['img_with_mask'] = image with transparent mask overlay
    """

    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        return plt.cm.get_cmap(name, n)


    def get_colored_mask(mask, n, cmap):
        """Given a BW mask, number of objects, and a LinearSegmentedColormap object, 
        returns a RGB mask.
        """
        color_mask = np.array([cmap(i) for i in mask])
        return np.array(color_mask)



    validate_image_with_mask(x, c)
    im = x['img']
    mask = x['mask']
    
    num_objects = np.unique(mask).shape[0]
    cmap = get_cmap(num_objects)
    mask_rgb = get_colored_mask(mask,num_objects,cmap)

    plt.imshow(im)
    plt.imshow(mask_rgb, alpha=transparency)
    plt.subplots_adjust(0,0,1,1)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = Image.open(buffer_)
    x['img_with_mask'] = np.array(image)
    buffer_.close()
    return x


>>>>>>> waldo-seg/master

def visualize_polygons(x):
    """This function accepts an object x that should represent an image with
       polygonal objects and it modifies the image to superimpose the edges of
       the polygon on it.
       This function returns None; it modifies x in-place.
    """
    validate_image_with_objects(x)
    # ... do something, modifying x somehow
    return None
