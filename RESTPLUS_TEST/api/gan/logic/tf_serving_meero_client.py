from __future__ import print_function

import os
import operator
import logging
import settings
import utils
import tensorflow as tf
import numpy as np
from PIL import Image

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


log = logging.getLogger(__name__)

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

def __get_tf_server_connection_params__():
    '''
    Returns connection parameters to TensorFlow Server

    :return: Tuple of TF server name and server port
    '''
    server_name = utils.get_env_var_setting('TF_SERVER_NAME', settings.DEFAULT_TF_SERVER_NAME)
    server_port = utils.get_env_var_setting('TF_SERVER_PORT', settings.DEFAULT_TF_SERVER_PORT)

    return server_name, server_port

def __create_prediction_request__(image):
    '''
    Creates prediction request to TensorFlow server for GAN model

    :param: Byte array, image for prediction
    :return: PredictRequest object
    '''
    # create predict request
    request = predict_pb2.PredictRequest()

    # Call GAN model to make prediction on the image
    request.model_spec.name = settings.GAN_MODEL_NAME
    request.model_spec.signature_name = settings.GAN_MODEL_SIGNATURE_NAME
    request.inputs[settings.GAN_MODEL_INPUTS_KEY].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1]))

    return request

def __open_tf_server_channel__(server_name, server_port):
    '''
    Opens channel to TensorFlow server for requests

    :param server_name: String, server name (localhost, IP address)
    :param server_port: String, server port
    :return: Channel stub
    '''
    channel = implementations.insecure_channel(
        server_name,
        int(server_port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    return stub

def __make_prediction_and_prepare_results__(stub, request):
    '''
    Sends Predict request over a channel stub to TensorFlow server

    :param stub: Channel stub
    :param request: PredictRequest object
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    result = stub.Predict(request, 60.0)  # 60 secs timeout


    # num_detections=	int(result.outputs['num_detections'].float_val[0])
    # boxes = np.reshape(result.outputs['detection_boxes'].float_val,[num_detections,4])
    # classes = np.squeeze(result.outputs['detection_classes'].float_val).astype(np.int32)
    # scores = np.squeeze(result.outputs['detection_scores'].float_val)
    
    # if 'detection_masks' in result.outputs:
    #     instance_masks=np.array(result.outputs['detection_masks'].float_val)
    #     N = result.outputs['detection_masks'].tensor_shape.dim[1].size
    #     height = result.outputs['detection_masks'].tensor_shape.dim[2].size
    #     width = result.outputs['detection_masks'].tensor_shape.dim[3].size
    #     instance_masks=np.reshape(instance_masks, (N,height, width))
    #     # instance_masks=format_mask(instance,boxes,num_detections,image_size)
    # else :
    #     instance_masks = None

    # return num_detections,boxes,classes,scores,instance_masks
    return result

def make_prediction(image):
    '''
    Predict the house number on the image using GAN model

    :param image: Byte array, images for prediction
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    # get TensorFlow server connection parameters
    server_name, server_port = __get_tf_server_connection_params__()
    log.info('Connecting to TensorFlow server %s:%s', server_name, server_port)

    # open channel to tensorflow server
    stub = __open_tf_server_channel__(server_name, server_port)

    # create predict request
    request = __create_prediction_request__(image)

    # make prediction
    return __make_prediction_and_prepare_results__(stub, request)
