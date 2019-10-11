# Tensorflow Object Detection API
This repository is a guide to illustrate the different step to obtain a Tf_object detection model from a Rhymse dataset to the deployement of this.

# STEP:
* Install Tensorflow Object Detection API as documented in the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
	- compiling the configuration protobufs and setting up the Python environment
* Install pycococtools
* Convert Rhymes_dataset to Cocoformat
* Run a trainning process Locally
* Export the model To create models ready for serving using export_model.py
	- exported the GAN model as Protobuf 
* Create TF-Serving environment using Docker
* Deploying Object Detection Model with TensorFlow Serving
	- Build the container using the official docker image
	- copy the model in the container
	- commit the change to create a new docker image
	- push this image to dockerhub ==> TensorFlow Server who host the model
* deploy using Flask web application
	- create the client  that is able “talk” over TensorFlow Serving works on gRPC protocol (tf_serving_sicara_client)
	- use Flask as Web framework to host my TensorFlow client (api_meero.py)
	- Dockerize the Flask application (dockerfile)
* Use DOCKER COMPOSE to run-start TensorFlow server and Flask application together
* Test  type the address http://0.0.0.0:5000/
