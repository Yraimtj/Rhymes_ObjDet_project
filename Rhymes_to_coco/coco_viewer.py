import os
# import cv2
import json
import math
import numpy as np

from PIL import Image , ImageDraw , ImageFont
import argparse
import io



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--json_path',type=str,help='path to json annotations')
parser.add_argument('--image_path',type=str, help='path to image')
args = parser.parse_args()

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

with open(args.json_path) as json_data:
    data =json.load(json_data)
 
image_address = args.image_path
VAL_IMAGE =Image.open(image_address)
filename = os.path.splitext(os.path.basename(image_address))[0]
#print (type(filename.split('_')[2]))
for obj in data['annotations']:
    if obj['image_id']==int(filename.split('_')[-1]) :
        for i in range(len(obj['segmentation'])):
#           [455.98, 436.73, 58.57, 36.36]  : [xmin,ymin,w,h]
            draw_object = ImageDraw.Draw(VAL_IMAGE)
            draw_object.polygon(obj['segmentation'][i],outline="red")
            draw_object.rectangle([obj['bbox'][0],obj['bbox'][1],obj['bbox'][0]+obj['bbox'][2],obj['bbox'][1]+obj['bbox'][3]],fill=None, outline="green")
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 25) ## change front size of TEXT
            draw_object.text([obj['bbox'][0],obj['bbox'][1]],str(data['categories'][obj['category_id']]['name']),font=fnt)
#             draw_object.rectangle(person['bbox'],fill=None, outline=None)
# cv2.imshow('Object detector', VAL_IMAGE)
# cv2.waitKey(0)
VAL_IMAGE.show()
