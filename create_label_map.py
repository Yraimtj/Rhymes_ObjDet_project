import json
import pandas as pd
import argparse
import os

"""
Creat a label_map.pbtxt file for one rhymes dataset.
Example usage:
    python create_label_map.py --dataset_name=meero\
        --cat_file_dir=D:/TOJO/DATASCIENCE/Meero/Meero_cat_batch1.csv \
        --output_path=D:/MEERO/batch3_2/
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name',type=str,help='name of the project')
parser.add_argument('--cat_file_dir',type=str, help='path to the csv category file')
parser.add_argument('--output_path',type=str, help='path to save the labe map file')
args = parser.parse_args()

cat_file_dir = args.cat_file_dir
dataset_name = args.dataset_name
output_path = os.path.join(args.output_path,args.dataset_name)

file = pd.read_csv(cat_file_dir)
categories = file['category_image'].unique()
end = '\n'
s = ' '
class_map = {}
for ID, name in enumerate(categories):
    out = ''
    out += 'item' + s + '{' + end
    out += s*2 + 'id:' + ' ' + (str(ID+1)) + end
    out += s*2 + 'name:' + ' ' + '\'' + name + '\'' + end
    out += '}' + end*2
    

    with open(output_path + '_label_map.pbtxt', 'a') as f:
        f.write(out)
        
    class_map[name] = ID+1
print(class_map)
