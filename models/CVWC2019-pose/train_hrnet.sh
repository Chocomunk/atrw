#! /bin/bash

S3_BASE=s3://calvinandpogs-ee148/atrw/out/pose/hrnet/$(date +%m-%d-%y-%H-%M-%S)
HRN_BASE=CVWC2019-pose/deep-high-resolution-net.pytorch

# Clone HRNet
# 1
#pip install torch >= 1.0.0
#2
git clone https://github.com/wanghao14/CVWC2019-pose
#3
cd "$HRN_BASE"
pip install -r requirements.txt
#4
cd lib
make
#5
cd ../../
#rm -rf cocoapi 
#pip install -U scikit-image
#pip install -U cython
#pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
cd cocoapi/PythonAPI
python3 setup.py install --user
cd ../../
#6
cd deep-high-resolution-net.pytorch
mkdir output log
#7 downloaded manually
#pip install gdown
#gdown --id 1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC

# Copy training data
mkdir -p data/atrw/pose models
ln -s "$SM_CHANNEL_POSE" data/atrw/pose
ln -s "$SM_CHANNEL_MODELS" models

# Train
python tools/train.py --cfg expriments/atrw/w48_384x288.yaml -dont_show 2>&1 | tee train_output.txt

# Test
python tools/test.py --cfg expriments/atrw/w48_384x288.yaml 2>&1 | tee test_output.txt

# Copy data back
aws s3 cp output "$S3_BASE"/output/
aws s3 cp log "$S3_BASE"/log/
aws s3 cp train_output.txt "$S3_BASE"/
aws s3 cp test_output.txt "$S3_BASE"/