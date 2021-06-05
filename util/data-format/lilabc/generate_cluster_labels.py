import os
import boto3
import argparse
from pathlib import Path

BUCKET = "calvinandpogs-ee148"

def main():
    prefix = "lilabc/images/"
    out_prefix = "lilabc/annotations/labels-full/"
    
    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET)
    
    # Generate cluster labels
    print("Loading clusters...")
    clusters = {}
    for object_summary in bucket.objects.filter(Prefix=prefix):
        p = Path(object_summary.key[len(prefix):])
        cluster = p.parts[0]
        if not cluster in clusters:
            clusters[cluster] = []
        clusters[cluster].append(p.stem)
        
    # Write cluster labels to aws
    print("Writing labels...")
    for cluster, f_names in clusters.items():
        out_file = os.path.join(out_prefix, "%s.txt"%cluster)
        client.put_object(Body='\n'.join(f_names), Bucket=BUCKET, Key=out_file)
    
    
if __name__ == "__main__":
    main()