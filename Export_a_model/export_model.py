"""
Example Usage:
--------------
python export_model.py \
    --input_type encoded_image_string_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory\
    --model_version_id 1
"""

import tensorflow as tf

# Assuming object detection API is available for use
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import get_configs_from_pipeline_file
import object_detection.exporter_serve

flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')

flags.DEFINE_string('pipeline_config_path', None,
                    'Path to the pipeline.config File.')

flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_directory', None, 'Path to write outputs.')

flags.DEFINE_integer('model_version_id', 1, 'integer for the current version of the model')

tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS

def main():

	# Configuration for model to be exported
	config_pathname = FLAGS.pipeline_config_path

	# Input checkpoint for the model to be exported
	# Path to the directory which consists of the saved model on disk (see above)
	trained_model_dir = FLAGS.trained_checkpoint_prefix 

	# Create proto from model confguration
	configs = get_configs_from_pipeline_file(config_pathname)
	pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

	# Read .ckpt and .meta files from model directory
	#checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
	#input_checkpoint = checkpoint.model_checkpoint_path
	input_checkpoint=FLAGS.trained_checkpoint_prefix

	# Model Version
	model_version_id = FLAGS.model_version_id

	# Output Directory
	output_directory = FLAGS.output_directory +"/"+ str(model_version_id)

	# Export model for serving
	object_detection.exporter_serve.export_inference_graph(input_type=FLAGS.input_type,
	pipeline_config=pipeline_proto,trained_checkpoint_prefix=input_checkpoint,
	output_directory=output_directory)

if __name__ == '__main__':
	main()
