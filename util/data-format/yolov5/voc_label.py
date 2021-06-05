import xml.etree.ElementTree as ET
import pickle
import os
import boto3
import argparse
from os import listdir, getcwd
from os.path import join
import subprocess

sets=[
        ('2019','train'),
        ('2019','val')
        # ('2019','test')
    ]

classes= ["Tiger"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(annot, image_id):
    in_file = open('%s/Annotations/%s.xml'%(annot, image_id))
    out_file = open('yolov5/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
                
def main():
    parser = argparse.ArgumentParser(description='YOLO Training')
    
    parser.add_argument('--output-dir', type=str, default="s3://calvinandpogs-ee148/atrw/detection/annotations/yolov5/")
    
#     parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--annot', type=str, default=os.environ['SM_CHANNEL_ANNOT'])

    args = parser.parse_args()
    wd = getcwd()
    
    image_set_dir = os.path.join(root_dir, "ImageSets", "Main", "")
    
    if not os.path.exists(image_set_dir):
        os.makedirs(image_set_dir)
        
    if not os.path.exists('yolov5/labels/'):
        os.makedirs('yolov5/labels/')

    for year, image_set in sets:
        image_ids = open('%s/ImageSets/Main/%s.txt'%(args.annot, image_set)).read().strip().split()
        list_file = open(os.path.join(image_set_dir, "%s.txt"%(image_set)), 'w')
        for image_id in image_ids:
            list_file.write('./data/images/tiger/%s.jpg\n'%(image_id))
            convert_annotation(args.annot, image_id)
        list_file.close()
        
    for path in execute(["aws", "s3", "cp", "--recursive", "yolov5/", args.output_dir]):
        print(path, end="")


if __name__ == '__main__':
    main()
