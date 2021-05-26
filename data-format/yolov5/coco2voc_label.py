import os
import json
import argparse
import subprocess

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
#     x = (box[0] + box[1])/2.0 - 1
#     y = (box[2] + box[3])/2.0 - 1
#     w = box[1] - box[0]
#     h = box[3] - box[2]
    x, y, w, h = box
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
        
def coco_to_dict(json_file):
    annot_dict = {}
    with open(json_file) as f:
        coco = json.load(f)
        
    # Initialize each image entry
    for img_data in coco['images']:
        img_id = img_data['id']
        annot_dict[img_id] = {}
        annot_dict[img_id]['size'] = (img_data['width'], img_data['height'])
        annot_dict[img_id]['bboxes'] = []
        
    # Populate bbox annotations
    for annot in coco['annotations']:
        img_id = annot['image_id']
        
        bbox = convert(annot_dict[img_id]['size'], annot['bbox'])
        bbox_string = str(annot['category_id'] - 1) + " " + " ".join([str(a) for a in bbox]) + '\n'
        annot_dict[img_id]['bboxes'].append(bbox_string)
        
    return annot_dict

def write_annots(annot_dict, image_set, image_set_dir, labels_dir):
    with open(os.path.join(image_set_dir, "%s.txt"%(image_set)), 'w') as list_file:
        for img_id, img_data in annot_dict.items():
            img_name = "{:04d}".format(img_id)
            list_file.write('./data/images/tiger-test/%s.jpg\n'%(img_name))
            
            with open(os.path.join(labels_dir, "%s.txt"%(img_name)), 'w') as annot_file:
                annot_file.writelines(img_data['bboxes'])

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
    
    parser.add_argument('--output-dir', type=str, default="s3://calvinandpogs-ee148/atrw/detection/annotations/yolov5-test/")
    
    parser.add_argument('--annot', type=str, default='test-annot/detect.json')

    args = parser.parse_args()
    
    image_set = 'test'
    image_set_dir = 'yolov5/ImageSets/'
    labels_dir = 'yolov5/labels/'
    
    if not os.path.exists(image_set_dir):
        os.makedirs(image_set_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    annots = coco_to_dict(args.annot)
    write_annots(annots, image_set, image_set_dir, labels_dir)
        
    for path in execute(["aws", "s3", "cp", "--recursive", "yolov5/", args.output_dir]):
        print(path, end="")


if __name__ == '__main__':
    main()