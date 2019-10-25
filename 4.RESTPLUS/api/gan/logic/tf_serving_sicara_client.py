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
from api.gan.logic.utils.fonction_util import format_mask, load_image_into_numpy_array




log = logging.getLogger(__name__)

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

def __make_prediction_and_prepare_results__(stub, request, image_path):
    '''
    Sends Predict request over a channel stub to TensorFlow server

    :param stub: Channel stub
    :param request: PredictRequest object
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    result = stub.Predict(request, 60.0)  # 60 secs timeout

    image = Image.open(image_path).convert("RGB")
    image_np = load_image_into_numpy_array(image)

    num_detections=	int(result.outputs['num_detections'].float_val[0])
    boxes_w=result.outputs['detection_boxes'].tensor_shape.dim[1].size
    boxes = np.reshape(result.outputs['detection_boxes'].float_val,[boxes_w,4])
    classes = np.squeeze(result.outputs['detection_classes'].float_val).astype(np.int32)
    scores = np.squeeze(result.outputs['detection_scores'].float_val)
    
    image_size=image_np.shape

    if 'detection_masks' in result.outputs:
        instance_masks=np.array(result.outputs['detection_masks'].float_val)
        N = result.outputs['detection_masks'].tensor_shape.dim[1].size
        height = result.outputs['detection_masks'].tensor_shape.dim[2].size
        width = result.outputs['detection_masks'].tensor_shape.dim[3].size
        instance=np.reshape(instance_masks, (N,height, width))
        instance_masks=format_mask(instance,boxes,num_detections,image_size)
    else :
        instance_masks = None

    return num_detections,boxes,classes,scores,instance_masks,image_np
    # return result

def make_prediction(image,image_path):
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
    return __make_prediction_and_prepare_results__(stub, request, image_path)
