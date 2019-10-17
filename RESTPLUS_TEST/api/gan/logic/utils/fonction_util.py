from __future__ import print_function

import os
import operator

import numpy as np
from PIL import Image
from skimage import measure


def format_mask(detection_masks, detection_boxes, N, image_size):
    """
    Format the m*m detection soft masks as full size binary masks. 

    Args:
        detection_masks (np.array): of size N * m * m
        detection_boxes (np.array): of size N * 4 with the normalized bow coordinates.
            Coordinates are written as [y_min, x_min, y_max, x_max]
        N (int): number of detections in the image
        image_size (tuple(int))

    Returns:
        detection_masks (np.array): of size N * H * W  where H and W are the image Height and Width.
    
    """
    (height, width, _) = image_size
    output_masks = np.zeros((N, image_size[0], image_size[1]))
    # Process the masks related to the N objects detected in the image
    for i in range(N):
        normalized_mask = detection_masks[i].astype(np.float32)
        normalized_mask = Image.fromarray(normalized_mask, 'F')

        # Boxes are expressed with 4 scalars - normalized coordinates [y_min, x_min, y_max, x_max]
        [y_min, x_min, y_max, x_max] = detection_boxes[i]

        # Compute absolute boundary of box
        box_size = (int((x_max - x_min) * width), int((y_max - y_min) * height)) 

        # Resize the mask to the box size using LANCZOS appoximation
        resized_mask = normalized_mask.resize(box_size, Image.LANCZOS)
        
        # Convert back to array
        resized_mask = np.array(resized_mask).astype(np.float32)

        # Binarize the image by using a fixed threshold
        binary_mask_box = np.zeros(resized_mask.shape)
        thresh = 0.5
        (h, w) = resized_mask.shape

        for k in range(h):
            for j in range(w):
                if resized_mask[k][j] >= thresh:
                    binary_mask_box[k][j] = 1

        binary_mask_box = binary_mask_box.astype(np.uint8)

        # Replace the mask in the context of the original image size
        binary_mask = np.zeros((height, width))
        
        x_min_at_scale = int(x_min * width)
        y_min_at_scale = int(y_min * height)

        d_x = int((x_max - x_min) * width)
        d_y = int((y_max - y_min) * height)

        for x in range(d_x):
            for y in range(d_y):
                binary_mask[y_min_at_scale + y][x_min_at_scale + x] = binary_mask_box[y][x] 
        
        # Update the masks array
        output_masks[i][:][:] = binary_mask

    # Cast mask as integer
    output_masks = output_masks.astype(np.uint8)
    return output_masks

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def mask_to_polyg(ground_truth_binary_mask):
    contours = measure.find_contours(ground_truth_binary_mask,0.8)
    liste = contours[0]
    polyg=[]
    for i in range (len(contours[0])):
        if i%10==0:
            y=int(liste[i][0])
            x=int(liste[i][1])

            polyg.append([x,y])
            
    return polyg

def convert_bbox (image_path,bbox):
    im = Image.open(image_path)
    ymin,xmin,ymax,xmax=bbox
    im_width, im_height = im.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    return left, right, top, bottom
