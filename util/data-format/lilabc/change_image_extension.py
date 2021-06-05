import os
import boto3
import argparse
from pathlib import Path

BUCKET = "calvinandpogs-ee148"

def main():
    root = "lilabc/"
    out_prefix = "lilabc/images/"
    in_prefix = "lilabc/cluster-images/"
    
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET)
    
    print("Renaming image extensions...")
    for object_summary in bucket.objects.filter(Prefix=root):
        p = Path(object_summary.key)
        if p.suffix == ".JPG":
            old_file = os.path.join(BUCKET, object_summary.key)
            new_file = str(p.with_suffix(".jpg"))
            
            print('Renaming: "{0}" to "{1}"'.format(old_file, new_file))
            
            s3.Object(BUCKET, new_file).copy_from(CopySource=old_file)
            s3.Object(BUCKET, object_summary.key).delete()
    
    
if __name__ == "__main__":
    main()