import io , os
import json

from flask import request
from flask_restplus import Resource
from api.restplus import api
from api.gan.logic.tf_serving_client import make_prediction
from werkzeug.datastructures import FileStorage
from werkzeug import secure_filename


# create dedicated namespace for GAN client
ns = api.namespace('gan_client', description='Operations for GAN client')

# Flask-RestPlus specific parser for image uploading
UPLOAD_KEY = 'image'
UPLOAD_LOCATION = 'files'
upload_parser = api.parser()
upload_parser.add_argument(UPLOAD_KEY,
                           location=UPLOAD_LOCATION,
                           type=FileStorage,
                           required=True#,
                           #action='append'
                           )


@ns.route('/prediction')
class GanPrediction(Resource):
    @ns.doc(description='Predict the house number on the image using GAN model. ' +
            'Return 3 most probable digits with their probabilities',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    @ns.expect(upload_parser)
    def post(self):
        try:
        #final_result=[]
            image_list = request.files[UPLOAD_KEY]
            filename = secure_filename(image_list.filename)
            img_path=os.path.join(filename)
            image_list.save(img_path)
            # args = upload_parser.parse_args()
            ##print("\n \n " + str(image_list))
            # for image_file in image_list:
            ## image = io.BytesIO(image_list.read())
            # print((image))
            with open(img_path, 'rb') as f:
                data = f.read()
        except Exception as inst:
            return {'message': 'something wrong with incoming request. ' +
                               'Original message: {}'.format(inst)}, 400

        try:
            # results = make_prediction(image.read())
            results = make_prediction(data)
            results_json = [{'digit': res[0], 'probability': res[1]} for res in results]
            #final_result.append(results_json)
            return {'prediction_result': results_json}, 200  

        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500
