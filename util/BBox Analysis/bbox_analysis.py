#import os
#import argparse
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

s3 = boto3.resource('s3')
bucket_name = 'calvinandpogs-ee148'
bucket = s3.Bucket(bucket_name)

def calcPercentArea(path):
    '''
    Input a path to the YOLOv5 bounding box labels.
    Returns list of % area for all bounding boxes for all labels in given path.
    
    INPUT 
    path : (str) path to labels folder
           e.g., 'atrw/out/detection/yolov5/test-m/05-26-2021-23-12-19/runs/labels/'
    
    OUTPUT
    area_lst : (list of floats) % area of all bounding boxes
    '''
    area_lst = []
    
    for obj in bucket.objects.all():
        if obj.key[0: len(path)] == path:
            body = obj.get()['Body'].read().decode("utf-8")
            boxes = body.split('\n')[:-1]
            
            # calc  % area from yolov5 format: A% = (w/W)*(h/H)
            percent_area = [float(box.split(' ')[-2])*float(box.split(' ')[-1]) for box in boxes]
            
            area_lst = area_lst + percent_area
    return area_lst

def percentArea(dataSubsetName, truthPath, **kwargs):
    '''
    Plot bounding box % area distribution for different runs of YOLOv5 given data subsets
    
    INPUT
    dataSubsetName: (str) data subset for plot title
                    e.g., 'Test', 'Augmented Train', etc.
    kwargs:
        keyword: Model run label e.g., 'YOLOv5s'
        argument: yolov5 2 subdirectories e.g., 'test-m/05-26-2021-23-12-19'
        e.g., YOLOv5s=test-m/05-26-2021-23-12-19
        
    OUTPUT
    plot : (.png) 
    '''
    export_path = 'atrw/out/detection/yolov5/figures/'
    
    # label paths
    paths = {'Ground truth': truthPath}
    for k, v in kwargs.items():
        paths[k] = 'atrw/out/detection/yolov5/' + v + '/runs/labels/' 
    
    # init plot
    fig = plt.figure()
    plot_kwargs = dict(alpha=0.5, bins=25)
    
    for cat, path in paths.items():
        # plot distribution
        percentAreaList = calcPercentArea(path)
        percentAreaDf = pd.DataFrame({'% Area': pd.Series(percentAreaList)})
        plt.hist(percentAreaDf, **plot_kwargs, label=cat)
        
        # upload df to s3
        csv_name = dataSubsetName + ''.join(cat.split()) + '.csv'
        percentAreaDf.to_csv(csv_name)
        s3.meta.client.upload_file(csv_name, bucket_name, export_path + csv_name)
        
        print(csv_name, 'uploaded')
    
    # finish plot
    plt.gca().set(title=dataSubsetName + ' Set Bounding Box % Area Distribution', \
              xlabel='Bounding Box % Area of Image', ylabel='Frequency')
    plt.yscale('log')
    plt.legend();
    
    # upload image to s3
    img_name = 'PercentArea' + dataSubsetName + '_' + '_'.join(list(kwargs.keys())) + '.png'
    fig.savefig(img_name, dpi=500)
    export_path = 'atrw/out/detection/yolov5/figures/' + img_name
    s3.meta.client.upload_file(img_name, bucket_name, export_path)
    
#percentArea('Train', truthPath='atrw/detection/annotations/yolov5/labels/', 
#           YOLOv5s='test-s/train/05-29-2021-03-27-33', 
#            YOLOv5m='test-m/train/05-29-2021-03-55-05')

#percentArea('Test', truthPath='atrw/detection/annotations/yolov5-test/labels/', 
#           YOLOv5s='test-s/test/05-27-2021-07-47-09', 
#            YOLOv5m='test-m/test/05-26-2021-23-12-19')

#def main():
#    parser = argparse.ArgumentParser(description='Bounding Box analysis')
#    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                        help='input batch size for training (default: 64)')
    
#if __name__ == '__main__':
#    main()