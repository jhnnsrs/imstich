# pip install numpy tifffile matplotlib
# pip install ashlarUC2 arkitekt-next[blok,all]
# python test_ashlar_numpy.py
from arkitekt_next import register, easy, progress
import time
import numpy as np
from mikro_next.api.schema import (
    Image,

    from_array_like,
    create_stage,
    PartialAffineTransformationViewInput,
    AffineTransformationView,
    Stage,
)
import xarray as xr
import numpy as np
import os
import re
from datetime import datetime
import tifffile
import numpy as np
from ashlarUC2.scripts.ashlar import process_images
from typing import List
import skimage.data as sd
from api.mikro import get_stage_views

@register
def create_stichable_astronaut(
 tiles_x: int = 3, tiles_y: int = 3
) -> List[Image]:
    """ Create a set of images that can be stitched together to form an astronaut image
    
    Args:
        tiles_x (int): Number of tiles in x direction
        tiles_y (int): Number of tiles in y direction

    Returns:
        List[Image]: List of images that can be stitched together
    
    """

    # create a list of images with random data
    stage = create_stage(name="Astrounaut Stage")

    astronaut = sd.astronaut()

    array = xr.DataArray(astronaut, dims=["x", "y", "c"]).transpose("c", "x", "y")
    print(array.shape)


    pixel_size_x = 0.5
    pixel_size_y = 0.5
    pixel_size_z = 1.0


    tile_dim_x = array.sizes["x"] // tiles_x
    tile_dim_y = array.sizes["y"] // tiles_y



    images = []
    for pos_x in range(tiles_x):
        for pos_y in range(tiles_y):
            data = array[: , pos_x * tile_dim_x : (pos_x + 1) * tile_dim_x, pos_y * tile_dim_y : (pos_y + 1) * tile_dim_y]

            #
            affine_matrix = np.array(
                [
                    [pixel_size_x, 0, 0, pos_x * tile_dim_x * pixel_size_x],
                    [0, pixel_size_y, 0, pos_y * tile_dim_y * pixel_size_y],
                    [0, 0, pixel_size_z, 0],
                    [0, 0, 0, 1],
                ]
            )

            transform_view = PartialAffineTransformationViewInput(
                affine_matrix=affine_matrix, stage=stage
            )

            image = from_array_like(
                xr.DataArray(data, dims=["c", "x", "y"]), name=f"tile_{pos_x}_{pos_y}", transformation_views=[transform_view]
            )
            images.append(image)

    return images




@register
def stitch_stage(
    stage: Stage,
    flip_x: bool = False,
    flip_y: bool = False,
    maximum_shift_microns: float = 50,
    stitch_alpha: float = 0.01,
) -> Image:
    """Stitch a stage of images together

    Args:
        stage (Stage): Stage to stitch
        flip_x (bool, optional): Flip the images in x direction. Defaults to False.
        flip_y (bool, optional): Flip the images in y direction. Defaults to False.
        maximum_shift_microns (float, optional): Maximum shift in microns. Defaults to 50.
        stitch_alpha (float, optional): Stitch alpha. Defaults to 0.01.

    Returns:
        Image: Stitched image
    """
    

    expanded_stage = get_stage_views(stage.id)
    
    cyx_arrays = []


    position_list = []
    pixel_size = None


    total_images = len(expanded_stage.affine_views)

    for i, view in enumerate(expanded_stage.affine_views):

        current_progress = (i + 1) / total_images * 80
        progress(current_progress, f"Downloading image {i + 1} of {total_images}")

        image_x_pos = view.affine_matrix[0][3]
        image_y_pos = view.affine_matrix[1][3]
        image_z_pos = view.affine_matrix[2][3]

        pixel_size_x = view.affine_matrix[0][0]
        pixel_size_y = view.affine_matrix[1][1]
        pixel_size_z = view.affine_matrix[2][2]

        assert pixel_size_x == pixel_size_y, "Pixel size must be the same in x and y. Not true for image: " + image.name
        if pixel_size is None:
            pixel_size = pixel_size_x

        assert pixel_size == pixel_size_x, "Pixel size must be the same for all images. Not true for image: " + image.name

        position_list.append([image_x_pos, image_y_pos])
        cyx_arrays.append(view.image.data.sel(t=0, z=0).transpose(*"cyx").compute())

    # stack the images (tile, channel, y, x) -> (tile, channel, y, x)
    stack = np.stack(cyx_arrays, axis=0)

    # algorithm needs the image in dimensions [(tiles, colour, channels, height, width)] = [4, 1, 1, 256, 256]
    # position_list needs to be a list of xy positions in microns
    # adjust to dataset:
    # create a 2D list of xy positions

    progress(80, "Stitching tiles with ashlar..")

    # Process numpy arrays
    mImage = process_images(
        filepaths=[stack],
        output="ashlar_output_numpy.tif",
        align_channel=0,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_mosaic_x=False,
        flip_mosaic_y=False,
        output_channels=None,
        maximum_shift=maximum_shift_microns,
        stitch_alpha=stitch_alpha,
        maximum_error=None,
        filter_sigma=0,
        filename_format="cycle_{cycle}_channel_{channel}.tif",
        pyramid=False,
        tile_size=1024,
        ffp=None,
        dfp=None,
        barrel_correction=0,
        plates=False,
        quiet=False,
        position_list=position_list,
        pixel_size=pixel_size,
    )

    # read the stitched image
    mImage = tifffile.imread("ashlar_output_numpy.tif")

    # create a new image object



    image = xr.DataArray(
        mImage,
        dims=["y", "x"],
    ) 

    upload = from_array_like(image, "ashlarstitch")

    os.remove("ashlar_output_numpy.tif")

    return upload



@register
def create_stitchable_images(
    tile_dim_x: int = 1000, tile_dim_y: int = 1000, tiles_x: int = 3, tiles_y: int = 3
) -> List[Image]:
    """ Create a set of images that can be stitched together

    Creates a set of images with random data that can be stitched together
    (mostly for testing purposes, not very useful in practice)

    Args:
        tile_dim_x (int): Tile dimension in x direction
        tile_dim_y (int): Tile dimension in y direction
        tiles_x (int): Number of tiles in x direction
        tiles_y (int): Number of tiles in y direction

    Returns:
        List[Image]: List of images that can be stitched together

    """

    # create a list of images with random data
    stage = create_stage(name="Stich")

    pixel_size_x = 0.5
    pixel_size_y = 0.5
    pixel_size_z = 1.0

    images = []
    for pos_x in range(tiles_x):
        for pos_y in range(tiles_y):
            data = np.random.random((tile_dim_x, tile_dim_y))

            #
            affine_matrix = np.array(
                [
                    [pixel_size_x, 0, 0, pos_x * tile_dim_x * pixel_size_x],
                    [0, pixel_size_y, 0, pos_y * tile_dim_y * pixel_size_y],
                    [0, 0, pixel_size_z, 0],
                    [0, 0, 0, 1],
                ]
            )

            transform_view = PartialAffineTransformationViewInput(
                affine_matrix=affine_matrix, stage=stage
            )

            image = from_array_like(
                data, name=f"tile_{pos_x}_{pos_y}", transformation_views=[transform_view]
            )
            images.append(image)

    return images


@register
def stitch2D(
    images: List[Image],
    flip_x: bool = False,
    flip_y: bool = False,
    maximum_shift_microns: float = 50,
    stitch_alpha: float = 0.01,
) -> Image:
    """Stitch a list of images together

    Stitches a list of images together using the ashlar algorithm

    Args:
        images (List[Image]): List of images to stitch
        flip_x (bool, optional): Flip the images in x direction. Defaults to False.
        flip_y (bool, optional): Flip the images in y direction. Defaults to False.
        maximum_shift_microns (float, optional): Maximum shift in microns. Defaults to 50.
        stitch_alpha (float, optional): Stitch alpha. Defaults to 0.01.

    Returns:
        Image: Stitched image
    """
    

    assert len(images) >= 4, "Needs at least 4 images to stitch"

    cyx_arrays = [image.data.sel(t=0, z=0).transpose(*"cyx") for image in images]
    shapes = [cyx_array.shape for cyx_array in cyx_arrays]

    # TODO: do we need to check for the same shape?
    assert len(set(shapes)) == 1, "All images must have the same shape"


    position_list = []
    pixel_size = None

    for image in images:

        affine_views = [view for view in image.views if isinstance(view, AffineTransformationView)]
        assert len(affine_views) == 1, "Each image must have exactly one AffineTransformationView, not true for image: " + image.name

        view = affine_views[0]


        image_x_pos = view.affine_matrix[0][3]
        image_y_pos = view.affine_matrix[1][3]
        image_z_pos = view.affine_matrix[2][3]

        pixel_size_x = view.affine_matrix[0][0]
        pixel_size_y = view.affine_matrix[1][1]
        pixel_size_z = view.affine_matrix[2][2]

        assert pixel_size_x == pixel_size_y, "Pixel size must be the same in x and y. Not true for image: " + image.name
        if pixel_size is None:
            pixel_size = pixel_size_x

        assert pixel_size == pixel_size_x, "Pixel size must be the same for all images. Not true for image: " + image.name

        position_list.append([image_x_pos, image_y_pos])



    # stack the images (tile, channel, y, x) -> (tile, channel, y, x)
    stack = np.stack(cyx_arrays, axis=0)

    # algorithm needs the image in dimensions [(tiles, colour, channels, height, width)] = [4, 1, 1, 256, 256]
    # position_list needs to be a list of xy positions in microns
    # adjust to dataset:
    # create a 2D list of xy positions

    print("Stitching tiles with ashlar..")

    # Process numpy arrays
    mImage = process_images(
        filepaths=[stack],
        output="ashlar_output_numpy.tif",
        align_channel=0,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_mosaic_x=False,
        flip_mosaic_y=False,
        output_channels=None,
        maximum_shift=maximum_shift_microns,
        stitch_alpha=stitch_alpha,
        maximum_error=None,
        filter_sigma=0,
        filename_format="cycle_{cycle}_channel_{channel}.tif",
        pyramid=False,
        tile_size=1024,
        ffp=None,
        dfp=None,
        barrel_correction=0,
        plates=False,
        quiet=False,
        position_list=position_list,
        pixel_size=pixel_size,
    )

    # read the stitched image
    mImage = tifffile.imread("ashlar_output_numpy.tif")

    # create a new image object

    image = xr.DataArray(
        mImage,
        dims=["y", "x"],
    ) 


    upload = from_array_like(image, "ashlarstitch")

    os.remove("ashlar_output_numpy.tif")

    return upload