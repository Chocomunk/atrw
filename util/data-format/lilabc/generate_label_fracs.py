import os
import math
import boto3
import argparse
from pathlib import Path

BUCKET = "calvinandpogs-ee148"

def main():
    prefix = "lilabc/annotations/labels-full/"
    out_prefix = "lilabc/annotations/labels-%d/"
    
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET)
    
    total_clusters = 60
    fracs = [2, 4, 8, 16]
    
    for frac in fracs:
        split = math.ceil(float(total_clusters) / frac) 
        out_folder = out_prefix % frac
        
        print("Generating split for: %d..." % frac)
        for cluster in range(split):
            fname = "cluster_%d.txt" % cluster
            new_file = os.path.join(out_folder, fname)
            old_file = os.path.join(BUCKET, prefix, fname)
            
            s3.Object(BUCKET, new_file).copy_from(CopySource=old_file)
            
    
if __name__ == "__main__":
    main()