#!/usr/bin/env python

import pandas as pd
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import urllib.request
import io
import json
import ast
import os

import argparse


import collections
import datetime
import glob
import os.path as osp
import sys
import math

#import labelme

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)

from sklearn.model_selection import train_test_split

"""
Convert dataset from RHYMES to coco format.
# It generates:
#   - data_dataset_coco/JPEGImages
#   - data_dataset_coco/annotations.json
Example usage:
    python Rhymes2Coco_format.py --name=meero \
        --csv_data=D:/DATASCIENCE/Meero/Meero_data_batch_1_v2.csv \
        --csv_cat=D:/DATASCIENCE/Meero/Meero_cat_bacth1.csv \
        --output_dir=D:/DATASCIENCE/Meero/data_dataset_coco/ \
        --labels=D:/DATASCIENCE/Meero/labels.txt \
        --test_size=0.2
"""

def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--name',type=str,help='name of the project')
    parser.add_argument('--csv_data',type=str, help='path to csv from rhymes')
    parser.add_argument('--csv_cat',type=str, help='path to csv containing category labels')
    parser.add_argument('--output_dir',type=str,
                        help='path to save annotation.json output and image')
    parser.add_argument('--labels', help='labels file', required=True)
    parser.add_argument('--test_size',help='percentage of test size', required=True)
    args = parser.parse_args()
    
    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir,'Train','JPEGImages'))
    os.makedirs(osp.join(args.output_dir,'Val','JPEGImages'))
    print('Creating dataset:', args.output_dir)

    # convert dataset to dataframe
    df1 = pd.read_csv(args.csv_data)
    df2 = pd.read_csv(args.csv_cat)
    
    print('\n\nNumber of input is {}\n\n'.format(len(df1)))
    now = datetime.datetime.now()

    ## split the dataset in 2 parts
    print(f'\n\nsplit the dataset in 2 parts\n\n')
    train, test = train_test_split(df1, test_size=float(args.test_size))
    train =train.reset_index(drop=True)  # rest index and drop old index
    test = test.reset_index(drop=True)   # rest index and drop old index
    dataset = [train, test]
        
    counter=0
    
    for sample in dataset:
                
        output_content = sample['output_content']  
        cdn_url=sample['cdn_file_url']
        cat=df2['category_image']
        cat_list = cat.values.tolist()
        # print(type(cat_list[0]))

        coco_data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
            ),
            licenses=[dict(
                url=None,
                id=0,
                name=None,
            )],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type='instances',
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        class_name_to_id = {}
        for i, line in enumerate(open(args.labels).readlines()):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            if class_id == -1:
                assert class_name == '__ignore__'
                continue
            class_name_to_id[class_name] = class_id
            coco_data['categories'].append(dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            ))
        if counter == 0:
            print(f'\n\nProcessing Train data\n\n')
            json_save_path = osp.join(args.output_dir, 'Train')
            sample_name=args.name + '_' + 'train2019'
        else:
            print(f'\n\nProcessing Val data\n\n')
            json_save_path = osp.join(args.output_dir, 'Val')
            sample_name=args.name + '_' + 'val2019'

        counter += 1   # counter for sample_name

        out_ann_file = osp.join(json_save_path, 'annotations.json')


        for i in range(len(output_content)): #for i in range(len(output_content)):
            data = dict(imagePath=None, shapes = [],imageData=None,lineColor=[0,255,0,128],fillColor=[255,0,0,128])
            a = eval(output_content[i])
            image_url = cdn_url[i]
            json_name =  sample_name+'_'+str(i) +".json"
            image_name = sample_name+'_'+ str(i) +".jpg"
            print(f'\nProcessing {image_name} data\n')
            data['imagePath'] = image_name
            image_id =i
            if 'identifications' in a:      # make the if statement here to avoid saving image and json without annotation

                for point in a['identifications']:
                    if point['type'] == 'boundingBox':
                        bbox = point['data']
                        polygone = [[bbox[0],bbox[1]],[bbox[0]+bbox[2],bbox[1]],[bbox[0]+bbox[2],bbox[1]+bbox[3]],[bbox[0],bbox[1]+bbox[3]]]
                        # print ('THE CAT IS {}'.format(type(point['attributes']['container'])))
                        if 'label' in point['attributes']:
                            if point['attributes']['label'] in cat_list:   # change container to label
                                category = point['attributes']['label']
                                data['shapes'].append(dict(
                                    line_color=None,points=polygone,
                                    fill_color=None,label=category
                                ))
                
                out_img_file = osp.join(
                    json_save_path, 'JPEGImages', image_name
                )
                
                with urllib.request.urlopen(image_url) as url:
                    f = io.BytesIO(url.read())
                img = Image.open(f)
                img.save(out_img_file,'JPEG')
                img_array = np.asarray(img)
                coco_data['images'].append(dict(
                    license=0,
                    url=image_url,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.size[1],
                    width=img.size[0],
                    date_captured=None,
                    id=image_id,
                ))
                
##                masks = {}                                     # for area
##                segmentations = collections.defaultdict(list)  # for segmentation
                for indx,shape in enumerate(data['shapes']):
                    nested_list = []
                    points = shape['points']
                    label = shape['label']
                    shape_type = shape.get('shape_type', None)
                    mask = shape_to_mask(
                        img_array.shape[:2], points, shape_type
                    )

##                    if label in masks:
##                        masks[label] = masks[label] | mask
##                    else:
##                        masks[label] = mask

                    points = np.asarray(points).flatten().tolist()
##                    segmentations[label].append(points)
                    nested_list.append(points)
                
##                for label, mask in masks.items():
                    cls_name = label.split('-')[0]
                    if cls_name not in class_name_to_id:
                        continue
                    cls_id = class_name_to_id[cls_name]

                    mask = np.asfortranarray(mask.astype(np.uint8))
                    mask = pycocotools.mask.encode(mask)
                    area = float(pycocotools.mask.area(mask))
                    bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                    coco_data['annotations'].append(dict(
                        id=len(coco_data['annotations']),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=nested_list,
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    ))

        with open(out_ann_file, 'w') as f:
            json.dump(coco_data, f,indent=4)




if __name__ == '__main__':
    main()
        
