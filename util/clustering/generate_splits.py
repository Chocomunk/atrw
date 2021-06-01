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
    parser.add_argument('--test-percent', type=float, default=0.20) # Note: percent of images     
    parser.add_argument('--val-percent', type=float, default=0.05)  # Note: percent of images

    parser.add_argument('--out-dir', type=str, default='./splits/')
    parser.add_argument('--input-data-dir', type=str)   
    # parser.add_argument('--output-s3', type=str, default='s3://calvinandpogs-ee148/atrw/detection/annotations/clusters')
    # parser.add_argument('--save-s3', type=lambda x: bool(strtobool(x)), default=True, help='En(dis)able uploading results to S3')

    args = parser.parse_args()

    # Create necessary folders
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    # Collect clusters
    # Assuming file format `label_{cluster_name}.txt`
    cluster_dict = {}
    with os.scandir(args.input_data_dir) as files:
        for file in files:
            if file.name.endswith('.txt'):
                with open(file) as f:
                    cluster = list(f)
                    cluster_images = []
                    for line in cluster:
                        image_num = line[:line.find('.')]
                        cluster_images.append(image_num)
                    file_name = str(file)
                    cluster_num = file_name[file_name.rfind('_')+1:file_name.rfind('.')]
                    cluster_dict[cluster_num] = cluster_images

    num_images = sum((len(v) for v in cluster_dict.values()))
    print("Collected {0} clusters containing {1} images ".format(len(cluster_dict), num_images))


    m = int(args.test_percent * num_images) # test set size
    print(m)

    cluster_lst = list(cluster_dict.keys())
    print(len(cluster_lst))
    random.shuffle(cluster_lst)
    
    # Create a test split of size m
    num_test_clusters = write_split(os.path.join(args.out_dir, "test.txt"), cluster_lst, cluster_dict, m)
    cluster_lst = cluster_lst[num_test_clusters:]

    # Create a validation split containing at least a certain number of images
    # based on a percentage of the total number of images
    print(len(cluster_lst))
    v = int(args.val_percent * num_images)
    print('v', v)
    num_val_clusters = write_split(os.path.join(args.out_dir, "val.txt"), cluster_lst, cluster_dict, v)


    # Create a train split with the remaining clusters
    cluster_lst = cluster_lst[num_val_clusters:]
    print(len(cluster_lst))

    write_split(os.path.join(args.out_dir, "train.txt"), cluster_lst, cluster_dict, num_images)

if __name__ == '__main__':
    main()
