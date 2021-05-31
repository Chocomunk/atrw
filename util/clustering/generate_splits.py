import os
import sys
import argparse
import contextlib
import random 
#from distutils.util import strtobool
from collections import deque
import itertools

def write_split(path, c_split, cluster_dict):
    with open(path, "w") as output:
        for c in c_split:
            for img_id in cluster_dict[c]:
                output.write('%s\n' % img_id)
def main():
    parser = argparse.ArgumentParser(description='Splits')
    parser.add_argument('--subset', type=int, default=-1) 
    parser.add_argument('--test-percent', type=float, default=0.20)     
    parser.add_argument('--val-percent', type=float, default=0.05)   

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
    #print(cluster_dict)

    # split clusters into splits
    k = len(cluster_dict) # total number of clusters
    m = int(args.test_percent * len(cluster_dict)) # test set size
    n = k - m # train/val set size
    print(k, m, n)

    cluster_lst = list(cluster_dict.keys())
    print(len(cluster_lst))

    random.shuffle(cluster_lst)
    
    write_split(os.path.join(args.out_dir, "test.txt"), cluster_lst[:m], cluster_dict)

    cluster_lst = cluster_lst[m:]
    print(len(cluster_lst))
    v = int(args.val_percent * num_images)
    print('v', v)
    val_size = 0
    trainval_cluster_count = 0
    with open(os.path.join(args.out_dir, "trainval.txt"), "w") as output:
            for c in cluster_lst:
                if val_size >= v:
                    break
                for img_id in cluster_dict[c]:
                    output.write('%s\n' % img_id)
                    val_size += 1

                trainval_cluster_count += 1
    print(trainval_cluster_count)
    cluster_lst = cluster_lst[trainval_cluster_count:]
    print(len(cluster_lst))

    write_split(os.path.join(args.out_dir, "train.txt"), cluster_lst, cluster_dict)

if __name__ == '__main__':
    main()
