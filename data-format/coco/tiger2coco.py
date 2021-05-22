import os
import argparse
import subprocess
from voc2coco import get_label2id, get_annpaths, convert_xmls_to_cocojson


image_sets = ["train", "val"]
        
    
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
                

def main():
    parser = argparse.ArgumentParser(description='ATRW to Coco conversion')
    
    parser.add_argument('--labels', type=str, default="./labels.txt", help='path to label list.')
    parser.add_argument('--ext', type=str, default='xml', help='additional extension of annotation file')
    parser.add_argument('--extract_num_from_imgid', action="store_true",
                        help='Extract image number from the image filename')
    
    parser.add_argument('--output-s3', type=str, default="s3://calvinandpogs-ee148/atrw/detection/annotations/coco/")
#     parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--annot', type=str, default=os.environ['SM_CHANNEL_ANNOT'])

    args = parser.parse_args()
    
    ann_dir = "{0}/Annotations/".format(args.annot)
    if not os.path.exists('coco_output/'):
        os.makedirs('coco_output/')
    
    for image_set in image_sets:
        ids_path = "{0}/ImageSets/Main/{1}.txt".format(args.annot, image_set)
        output = "coco_output/{0}.json".format(image_set)
        
        label2id = get_label2id(labels_path=args.labels)
        ann_paths = get_annpaths(
            ann_dir_path=ann_dir,
            ann_ids_path=ids_path,
            ext=args.ext,
            annpaths_list_path=None
        )
        convert_xmls_to_cocojson(
            annotation_paths=ann_paths,
            label2id=label2id,
            output_jsonpath=output,
            extract_num_from_imgid=args.extract_num_from_imgid
        )
        
    for path in execute(["aws", "s3", "cp", "--recursive", "coco_output/", args.output_s3]):
        print(path, end="")
        
    
if __name__ == '__main__':
    main()