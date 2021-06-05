import os
import boto3
import random
from pathlib import Path

BUCKET = "calvinandpogs-ee148"

def main():
    splits = list(range(10))
    prefix = "atrw/out/detection/yolov5/train-s/clustering/vgg16places/split%d/"
    out_prefix = "models/yolov5/train-s/clusters/split%d/"
    
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET)
    
    for split in splits:
        prefix_s = prefix % split
        prefix_o = out_prefix % split
        
        # Find all unique training runs
        unique_runs = set()
        for object_summary in bucket.objects.filter(Prefix=prefix_s):
            p = Path(object_summary.key[len(prefix_s):])
            unique_runs.add(p.parts[0])
        
        last_run = sorted(list(unique_runs))[-1]
        weights_path = os.path.join(prefix_s, last_run, "runs", "weights")
        for object_summary in bucket.objects.filter(Prefix=weights_path):
            p = Path(object_summary.key)
            new_file = os.path.join(prefix_o, p.name)
            old_file = os.path.join(BUCKET, str(p))
            
            print(new_file, old_file)
            
            s3.Object(BUCKET, new_file).copy_from(CopySource=old_file)
            
    
if __name__ == "__main__":
    main()