import numpy as np
from skimage.transform import hough_line, hough_line_peaks

from aicssegmentation.core.vessel import filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects  
from skimage.draw import polygon

# create a mask out of vertices
def create_mask_from_shapes(vertices_polygons, im_shape):

    '''
    Input:
    - vertices_polygons - list of polygons (coordinates of vertices)
    - im_shape - ize of the image to create
    Output:
    - label image of polygons
    '''

    mask = np.zeros(im_shape).astype('uint8')

    for i,poly in enumerate(vertices_polygons):

        # if drawing was in 3D
        if len(poly.shape) > 2:
            mask_coord = polygon(vertices_polygons[i][:,1],vertices_polygons[i][:,2],shape=im_shape)
        else:
            mask_coord = polygon(vertices_polygons[i][:,0],vertices_polygons[i][:,1],shape=im_shape)

        mask[mask_coord] = i+1

    return mask


def segment_actin_3D(image_actin):

    '''
    Function wrapping Allen Cell Segmenter algorithm for fibers.
    Note that it will segment differently single cells and entire field of view because of the normalization steps.
    Based on: https://github.com/AllenCell/aics-segmentation/blob/main/lookup_table_demo/playground_filament3d.ipynb 

    Input:
    image_actin - 3D image of actin 

    Output:
    image_actin_mask - 3D segmented fibers
    '''

    ################################
    ## PARAMETERS for this step ##
    intensity_scaling_param = [0]
    f3_param = [[1, 0.01]]
    ################################

    # intensity normalization
    struct_img = intensity_normalization(image_actin, scaling_param=intensity_scaling_param)

    # smoothing with edge preserving smoothing 
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    # segmentation
    image_actin_mask = filament_3d_wrapper(structure_img_smooth, f3_param)

    return image_actin_mask


def find_fibers_orientation(image_actin_mask_2D):

    '''
    Function that uses Hough transform to find a dominant orientation of fibers in a 2D image of single cells.

    Input:
    image_actin_mask_2D

    Output:
    dominant_flow - angle (radians) 
    '''

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image_actin_mask_2D, theta=tested_angles)

    _, angle_array, dist_array = hough_line_peaks(h, theta, d)

    dominant_flow = np.mean(angle_array[:4])

    return dominant_flow