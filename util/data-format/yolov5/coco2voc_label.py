import os
import json
import argparse

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    l, t, w, h = box        # top-left, (width, heigh)
    x = l + (w/2.0)
    y = t + (h/2.0)
    x1 = x*dw
    w1 = w*dw
    y1 = y*dh
    h1 = h*dh
    return (x1,y1,w1,h1), (int(x),int(y),w,h)    # Image Scaled, Native Scaling
        
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
        bbox_string = str(annot['category_id'] - 1) + " " + " ".join([str(a) for a in bbox[0]]) + '\n'
        annot_dict[img_id]['bboxes'].append(bbox_string)
        annot['bbox'] = bbox[1]    # Convert existing bbox format
#         annot['category_id'] = (1 - annot['category_id'])
        
    return annot_dict, coco

def write_annots(annot_dict, image_set, image_set_dir, labels_dir):
    with open(os.path.join(image_set_dir, "%s.txt"%(image_set)), 'w') as list_file:
        for img_id, img_data in annot_dict.items():
            img_name = "{:04d}".format(img_id)
            list_file.write('./data/images/tiger-test/%s.jpg\n'%(img_name))
            
            with open(os.path.join(labels_dir, "%s.txt"%(img_name)), 'w') as annot_file:
                annot_file.writelines(img_data['bboxes'])
                
def main():
    parser = argparse.ArgumentParser(description='YOLO Training')
    
    parser.add_argument('--output-dir', type=str, default="s3://calvinandpogs-ee148/atrw/detection/annotations/yolov5-test/")
    parser.add_argument('--annot', type=str, default='test-annot/detect.json')
    parser.add_argument('--no-s3', action='store_true', help='Disables uploading to s3')

    args = parser.parse_args()
    
    image_set = 'test'
    root_dir = 'yolov5'
    image_set_dir = os.path.join(root_dir, "ImageSets", "")
    labels_dir = os.path.join(root_dir, "labels", "")
    
    if not os.path.exists(image_set_dir):
        os.makedirs(image_set_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
        
    print("Converting annotations...")
    
    annots, coco_dict = coco_to_dict(args.annot)
    write_annots(annots, image_set, image_set_dir, labels_dir)
    
    with open(os.path.join(root_dir, "%s.json"%(image_set)), 'w') as new_coco:
        json.dump(coco_dict, new_coco)
    

    print("Conversion Complete!")
        
    if not args.no_s3:
        cmd = "aws s3 cp --recursive {0} {1} --exclude {0}/.ipynb_checkpoints/*".format(root_dir, args.output_dir)
        print("Executing: {0}".format(cmd))
        os.system("{0} >/dev/null".format(cmd))

    print("Done!")


if __name__ == '__main__':
    main()