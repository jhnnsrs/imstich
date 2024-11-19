# pip install numpy tifffile matplotlib
# pip install ashlarUC2 arkitekt-next[blok,all] 
# python test_ashlar_numpy.py
from arkitekt_next import register, easy
import time
import numpy as np
from mikro_next.api.schema import Image, from_array_like
import os
import re
from datetime import datetime
import tifffile
import numpy as np
from ashlarUC2.scripts.ashlar import process_images


@register 
def move_stage(axis: str="X", position:int=0):
    print("Moving the stage %s, at %i", axis, position)
    

@register
def append_world(hello: str) -> str:
    """Append World

    This function appends world to the input string

    Parameters
    ----------
    hello : str
        The input string

    Returns
    -------
    str
        {{hello}} World
    """ """"""
    return hello + " World"




@register
def stitch2D(pixel_size:float=0.5, position_list:np.ndarray=None, arrays:np.ndarray=None, flip_x:bool=False, flip_y:bool=False) -> Image:
    # algorithm needs the image in dimensions [(tiles, colour, channels, height, width)] = [4, 1, 1, 256, 256]
    # position_list needs to be a list of xy positions in microns
    # adjust to dataset:
    maximum_shift_microns = 50
    num_images = 4
    num_channels = 2
    height, width = 256, 256
    arrays = [np.random.rand(num_images, num_channels, height, width)]
    # create a 2D list of xy positions 
    position_list = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))*260.
    pixel_size = 0.5
    
    arrays = [np.expand_dims(np.array(arrays), axis=1)] # ensure channel is set
    position_list = np.array(position_list) # has to be a list

    print("Stitching tiles with ashlar..")

    # Process numpy arrays
    mImage = process_images(filepaths=arrays,
                    output='ashlar_output_numpy.tif',
                    align_channel=0,
                    flip_x=False,
                    flip_y=False,
                    flip_mosaic_x=False,
                    flip_mosaic_y=False,
                    output_channels=None,
                    maximum_shift=maximum_shift_microns,
                    stitch_alpha=0.01,
                    maximum_error=None,
                    filter_sigma=0,
                    filename_format='cycle_{cycle}_channel_{channel}.tif',
                    pyramid=False,
                    tile_size=1024,
                    ffp=None,
                    dfp=None,
                    barrel_correction=0,
                    plates=False,
                    quiet=False, 
                    position_list=position_list,
                    pixel_size=pixel_size)

    # check if data was read correctly
    mImage = np.random.random((1000,1000))
    return from_array_like(mImage, "ashlarstitch")


# This is the part that runs the server
# Everything must be registered before this function is called


# The easy function is a context manager as it will need to clean
# up the resources it uses when the context is exited (when the user stops the app)
# make sure to give your app a name, (and the url/ip of the arkitekt server) 
with easy("Stitcher", url="localhost") as e:

    # If you want to perform a request to the server before enabling the
    # provisioning loop you can do that within the context

    # from_array_like(np.random.rand(100, 100, 3) * 255, name="test")
    # would upload an image to the server on app start

    # e.run() will start the provisioning loop of this app
    # this will block the thread and keep the app running until the user
    # stops the app (keyboard interrupt)
    e.run()
