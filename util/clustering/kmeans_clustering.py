# Based on the blog post "How to cluster images based on visual similarity" 
# written by Gabe Flomo.


# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def extract_features(file, model):
    # load the image from the s3 bucket as a 224x224 array and convert from 
    # 'PIL.Image.Image' to numpy array because the VGG model expects the images
    # it receives to be 224x224 NumPy arrays.

    # object = s3.Object(bucket, images_path + '/' + file)
    # img = load_img(BytesIO(object.get()['Body'].read()), target_size=(224,224))
    img = load_img(file, target_size=(224,224))

    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster, groups, input_dir, output_dir):
    plt.figure(figsize = (25,25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    if len(files) > 30:
        print(f"cluster {cluster} has more tha 30 images!")
    # only allow up to 30 images to be shown at a time
    #     print(f"Clipping cluster size from {len(files)} to 30 for cluster " + str(cluster))
    #     files = files[:30]

    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1)
        img = load_img(input_dir + '/' + file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

    save_path = output_dir + '/view_clusters'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + '/view_cluster' + str(cluster) + '.png', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Clustering')

    parser.add_argument('--output-data-dir', type=str)
    parser.add_argument('--input-data-dir', type=str)   
    # raises keyerror locally
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--subset', type=int, default=-1)   
    parser.add_argument('--num-clusters', type=int, default=50)   

    parser.add_argument('--getbestc', type=bool, default=False)   

    args = parser.parse_args()

    
    #file_dir = os.getcwd()
    # change the working directory to the path where the images are located
    data_path = args.input_data_dir
    os.chdir(data_path)

    image_fnames = []
    # creates a ScandirIterator aliased as files
    print('------------ getting image names ------------')
    with os.scandir(data_path) as files:
      # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
              # adds only the image files to the image_fnames list
                image_fnames.append(file.name)
   
    print(str(len(image_fnames)) + ' images collected')

        
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    img_features = {}

    # p = r"CHANGE TO A LOCATION TO SAVE FEATURE VECTORS"

    # loop through each image in the dataset
    if args.subset != -1:
        subset_len = args.subset
    else:
        subset_len = len(image_fnames)

    for image in tqdm(image_fnames[:subset_len]):
        # try to extract the features and update the dictionary
        feat = extract_features(image,model)
        img_features[image] = feat
        
    
    # get a list of the filenames
    filenames = np.array(list(img_features.keys()))

    # get a list of just the features
    feat = np.array(list(img_features.values()))

    feat = feat.reshape(-1,4096)


    # # reduce the amount of dimensions in the feature vector
    print('------------ feature reduction ------------')
    pca = PCA(n_components=100, random_state=22)
    x = pca.fit_transform(feat)


    # this is just incase you want to see which value for k might be the best 
    # note: this didn't really work very well (gave best number of clusters = 5
    # which is obviously too little)
    if args.getbestc:
        sse = []
        list_k = list(range(5, subset_len))
        max_diff = 0
        prev_sse = 0
        best_k = 0
        attempts = 0

        print('------------ checking clusters ------------')

        # find the number of clusters that produces the greatest change in sse. 
        for k in tqdm(list_k):
            km = KMeans(n_clusters=k, random_state=22)
            km.fit(x)
            inertia = km.inertia_
            if abs(inertia - prev_sse) > max_diff:
                print('new best k : ' + str(k))
                max_diff = inertia 
                best_k = k 
                attempts = 0
            prev_see = inertia
            sse.append(inertia)
            attempts += 1

        print('best number of clusters: ' + str(best_k))

        # Plot sse against k
        print('------------ plotting sse against k ------------')
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse)
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        plt.show()
        plt.savefig(args.output_data_dir + '/cluster_analysis.png')

        k = best_k
    else:
        k = args.num_clusters

    # cluster feature vectors
    kmeans = KMeans(n_clusters=k, random_state=22)
    label = kmeans.fit_predict(x)
    kmeans.fit(x)
    
    print('------------ printing labels ------------')
    print(kmeans.labels_)
    #Getting unique labels
    
    u_labels = np.unique(kmeans.labels_)
    
    #plotting the results:
    
    for i in u_labels:
        plt.scatter(x[label == i , 0] , x[label == i , 1] , label = i)

    plt.savefig(args.output_data_dir + 'pca_clustering.png', bbox_inches='tight')
    plt.close()

    # holds the cluster id and the images { id: [images] }
    print('------------ getting groups ------------')

    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    print(groups)

    print('------------ creating label files ------------')


    for cluster in groups.keys(): 
        with open(f"{args.output_data_dir}/labels/label_{cluster}.txt", "w") as output:
            for file in groups[cluster]:
                output.write('%s\n' % file)



    print('------------ saving group images ------------')
    for cluster in groups.keys():
        if len(groups[cluster]) > 1:
            view_cluster(cluster, groups, args.input_data_dir, args.output_data_dir)


    print('DONE!')
    
if __name__ == '__main__':
    main()