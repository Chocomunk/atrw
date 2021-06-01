# Based on the blog post "How to cluster images based on visual similarity" 
# written by Gabe Flomo.

import os
import sys
import argparse
import contextlib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from distutils.util import strtobool

from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Hide stdout of noisy functions
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

    
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def extract_features(file_name, model):
    img = load_img(file_name, target_size=(224,224))

    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3) 
    imgx = preprocess_input(reshaped_img)
    return model.predict(imgx, use_multiprocessing=True)


def view_cluster(image_files, cluster_name, input_dir, output_dir, max_images=16):
    # Limit images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]

    # Plot each image in the cluster
    # TODO: Make subplot grid change with max_images
    plt.figure(figsize = (25,25))
    for index, file in enumerate(image_files):
        plt.subplot(4,4,index+1)
        img = load_img(os.path.join(input_dir, '%s.jpg'%(file)))
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

    save_path = os.path.join(output_dir, 'view_clusters')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'view_cluster{cluster_name}.png'), bbox_inches='tight')
    plt.close()
    
    
def find_num_clusters(max_clusters, feats, out_dir):
    sse = []
    list_k = list(range(5, max_clusters))
    max_diff = 0
    prev_sse = 0
    best_k = 0
    attempts = 0

    # find the number of clusters that produces the greatest change in sse. 
    for k in list_k:
        km = KMeans(n_clusters=k, random_state=22)
        km.fit(feats)
        inertia = km.inertia_
        if abs(inertia - prev_sse) > max_diff:
            max_diff = inertia 
            best_k = k 
            attempts = 0
        prev_see = inertia
        sse.append(inertia)
        attempts += 1

    print("Best number of clusters: {0}".format(best_k))

    # Plot sse against k
    print("Plotting SSE vs. num_clusters (k)...")
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse)
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.savefig(os.path.join(out_dir, 'plots', 'cluster_analysis.png'))
    plt.show()
    
    return best_k


def main():
    parser = argparse.ArgumentParser(description='Clustering')
    parser.add_argument('--subset', type=int, default=-1)   
    parser.add_argument('--num-clusters', type=int, default=50)   
    parser.add_argument('--getbestc', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--view-images', type=lambda x: bool(strtobool(x)), default=True)

    parser.add_argument('--out-dir', type=str, default='./clusters/')
    parser.add_argument('--input-data-dir', type=str, default=os.environ['SM_CHANNEL_IMAGES'])   
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-s3', type=str, default='s3://calvinandpogs-ee148/atrw/detection/annotations/clusters')
    parser.add_argument('--save-s3', type=lambda x: bool(strtobool(x)), default=True, help='En(dis)able uploading results to S3')

    args = parser.parse_args()
    
    # Create necessary folders
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(os.path.join(args.out_dir, 'plots')):
        os.makedirs(os.path.join(args.out_dir, 'plots'))
    if not os.path.exists(os.path.join(args.out_dir, 'labels')):
        os.makedirs(os.path.join(args.out_dir, 'labels'))
    
    # TODO: Stop filling the array once we hit `subset`
    # Get image names
    image_fnames = []
    with os.scandir(args.input_data_dir) as files:
        for file in files:
            if file.name.endswith('.jpg'):
                image_fnames.append(file.name)
   
    subset_len = len(image_fnames) if args.subset == -1 else args.subset
    print("Collected {0} images".format(subset_len))

        
    # Extract features for each image
    print("Loading feature model...")
    with nostdout():
        model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    img_features = {}

    for image in image_fnames[:subset_len]:
        feat = extract_features(os.path.join(args.input_data_dir, image),model)
        img_features[os.path.splitext(image)[0]] = feat    # Remove extension
        
    feats = np.array(list(img_features.values())).reshape(-1,4096)
    
    print("Clustering features...")
    
    # Dimensionality reduction
    x = PCA(n_components=100, random_state=22).fit_transform(feats)

    # Clustering
    k = find_num_clusters(subset_len, x, args.out_dir) if args.getbestc else args.num_clusters
    kmeans = KMeans(n_clusters=k, random_state=22)
    label = kmeans.fit_predict(x)
    kmeans.fit(x)
    
    # Full-reduce down to 2 dimensions
    y = PCA(n_components=2, random_state=22).fit_transform(x)
    for i in np.unique(kmeans.labels_):
        plt.scatter(y[label == i , 0], y[label == i , 1], s=5, label=i)
    plt.title('Train-Image Feature Clusters')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(os.path.join(args.out_dir, 'plots', 'clustering_scatter.png'), bbox_inches='tight')
    plt.close()

    # Save cluster groups
    groups = {}
    for file, cluster in zip(img_features.keys(),kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    for cluster in groups.keys(): 
        with open(os.path.join(args.out_dir, "labels", f"label_{cluster}.txt"), "w") as output:
            for file in groups[cluster]:
                output.write('%s\n' % file)

    # Render Clusters
    if args.view_images:
        print("Rendering Cluster Examples...")
        for cluster, images in groups.items():
            if len(images) > 1:
                view_cluster(images, cluster, args.input_data_dir, args.out_dir)

    # Upload to S3
    if args.save_s3:
        print("Executing: aws s3 cp --recursive {} {}".format(args.out_dir, args.output_s3))
        os.system("aws s3 cp --recursive {} {} >/dev/null".format(args.out_dir, args.output_s3))
    
    
if __name__ == '__main__':
    main()