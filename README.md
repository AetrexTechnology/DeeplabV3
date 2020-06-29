# DeepLab: Deep Labelling for Semantic Image Segmentation

DeepLab is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g., person, dog, cat and so on)
to every pixel in the input image. Current implementation includes the following
features:


 DeepLabv3+ [4]: We extend DeepLabv3 to include a simple yet effective
    decoder module to refine the segmentation results especially along object
    boundaries. Furthermore, in this encoder-decoder structure one can
    arbitrarily control the resolution of extracted encoder features by atrous
    convolution to trade-off precision and runtime.

Please follow below steps to run the model(DeepLabv3)

Deploying deeplabv3 model steps:

1) install python3 and pip3

2) pip3 install tensorflow==1.15

3) pip3 install Pillow

4) pip3 install tf_slim. 


5) click on those 2 links and clone those 2 directories below:
https://github.com/heaversm/deeplab-training         ..... models-master

https://github.com/tensorflow/models    .....deeplab/master 

6) Clone all the dependencies from Requirements.txt which is attached in the github main link. 

7) now In models-master copy slim folder from tensorflow/models and in deeplab folder replace this file input_preprocess.py from tensorflow/models file input_preprocess.py.

8) now remove all aother files in models-master and put models/research/eval-pqr.sh into the TensorFlow models/research directory.
models/research/deeplab and models/research/slim


10) run this command everytime before ruuning the python script on terminal and folder structure will be models/research:       
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

and then run         python3 deeplab/model_test.py        in the same path.


11) dataset creation :  a) ImageSets      (train(90%) and validation test and in train val data(put all data from JPEGImages))
b) JPEGImages : image with coin and foot
c)SegmentationClass :  maskimage

12) run the convert_rgb_to_index.py.py file in /home/sonu/Downloads/models-master/research/deeplab/datasets ......do changes in image shape only ....it will create 
SegmentationClassRawa dataset in same folder of SegmentationClass.

13) run the build_pqr_data.py file in /home/sonu/Downloads/models-master/research/deeplab/dataset.....do changes in path of all file and .png to jpg
        It will create tfrecord folder in same path /home/sonu/Downloads/models-master/research/deeplab/dataset.

14) do changes in train-pqr.sh file and run command sh train-pqr.sh or search error in train.py file. 


for checking the results if model is running...........goto.....PQR...exp.....train_ckpt folder

