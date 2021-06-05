import os
import sys
import argparse
import random 
#from distutils.util import strtobool
from collections import deque
import itertools

def write_split(path, c_split, cluster_dict, split_size):
    curr_size = 0
    cluster_count = 0

    with open(path, "w") as output:
        for c in c_split:
            if curr_size >= split_size:
                break
            for img_id in cluster_dict[c]:
                output.write('%s\n' % img_id)
                curr_size += 1
            cluster_count += 1
    return cluster_count


def main():
    parser = argparse.ArgumentParser(description='Splits')
    parser.add_argument('--subset', type=int, default=-1) 
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--test-percent', type=float, default=0.10) # Note: percent of images     
    parser.add_argument('--val-percent', type=float, default=0.10)  # Note: percent of images

    parser.add_argument('--out-dir', type=str, default='./splits/')
    parser.add_argument('--input-dir', type=str, default=os.environ['SM_CHANNEL_CLUSTERS'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-s3', type=str, default='s3://calvinandpogs-ee148/atrw/detection/annotations/cluster-splits/')
    parser.add_argument('--save-s3', type=lambda x: bool(strtobool(x)), default=True, help='En(dis)able uploading results to S3')

    args = parser.parse_args()
    sub_dir = os.path.join(args.out_dir, "ImageSets/clustering/")

    # Collect clusters
    # Assuming file format `label_{cluster_name}.txt`
    cluster_dict = {}
    with os.scandir(args.input_dir) as files:
        for file in files:
            file_name = str(file)
            if file.name.endswith('.txt'):
                with open(file) as f:
                    cluster_images = ['./data/images/tiger/%s.jpg'%(image_id.rstrip()) for image_id in list(f)]
                    cluster_num = file_name[file_name.rfind('_')+1:file_name.rfind('.')]
                    cluster_dict[cluster_num] = cluster_images

    num_images = sum((len(v) for v in cluster_dict.values()))
    print("Collected {0} clusters containing {1} images ".format(len(cluster_dict), num_images))

    for i in range(args.num_splits):
        # Create necessary folders
        split_dir = os.path.join(sub_dir, "split{0}".format(i))
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        # Shuffle the clusters for sampling
        cluster_lst = list(cluster_dict.keys())
        random.shuffle(cluster_lst)

        # Save the test split
        m = int(args.test_percent * num_images) # test set size
        num_test_clusters = write_split(os.path.join(split_dir, "test.txt"), cluster_lst, cluster_dict, m)
        cluster_lst = cluster_lst[num_test_clusters:]

        # Save the val split
        v = int(args.val_percent * num_images)
        num_val_clusters = write_split(os.path.join(split_dir, "val.txt"), cluster_lst, cluster_dict, v)
        cluster_lst = cluster_lst[num_val_clusters:]

        # Save the remaining data as the train split
        write_split(os.path.join(split_dir, "train.txt"), cluster_lst, cluster_dict, num_images)

    # Upload to S3
    if args.save_s3:
        print("Executing: aws s3 cp --recursive {} {}".format(args.out_dir, args.output_s3))
        os.system("aws s3 cp --recursive {} {} >/dev/null".format(args.out_dir, args.output_s3))

if __name__ == '__main__':
    main()
