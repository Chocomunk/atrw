import os
import boto3
import argparse
from pathlib import Path

BUCKET = "calvinandpogs-ee148"

def main():
    prefix = "lilabc/annotations/"
    out_prefix = "lilabc/images/"
    in_prefix = "lilabc/cluster-images/"
    
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET)
    
    # Find image-sets
    print("Loading image sets...")
    image_sets = {}
    for object_summary in bucket.objects.filter(Prefix=prefix):
        p = Path(object_summary.key[len(prefix):])
        image_set = p.parts[0]
        if not image_set in image_sets:
            image_sets[image_set] = []
        image_sets[image_set].append(p.stem)
        
    # Copy cluster images
    print("Copying images...")
    for image_set, clusters in image_sets.items():
        out_folder = os.path.join(out_prefix, image_set)
        
        print("Copying images for: {0}...".format(image_set))
        for cluster in clusters:
            cluster_folder = os.path.join(in_prefix, cluster, '')
            for object_summary in bucket.objects.filter(Prefix=cluster_folder):
                p = Path(object_summary.key)
                old_file = os.path.join(BUCKET, object_summary.key)
                new_file = os.path.join(out_folder, p.name)
            
                s3.Object(BUCKET, new_file).copy_from(CopySource=old_file)
    
    
if __name__ == "__main__":
    main()