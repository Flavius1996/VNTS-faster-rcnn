### Vietnam Traffic Signs Detection and Classification
This module is an extension of [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) to detect and classify [Vietnam Traffic Signs](https://drive.google.com/open?id=0B9hMAZTpHpyCclFwT2NFWTRYSjg).
For installation, I modified the original Faster-RCNN [rbgirshick](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) and [sridhar912](https://github.com/sridhar912/tsr-py-faster-rcnn/README.md) file to adapt changes for run this module. Please check below for license and citation information.

View step-by-step of my modification at:
	https://docs.google.com/document/d/1KnBSpKmxlk21EGc_wvxeYGF7Cd-LIoZHF81NKm9Ze0U/edit#heading=h.hejksbigjuke


### Contents
1. [Requirements: software & hardware](#requirements-software-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software & hardware

More details at https://github.com/rbgirshick/py-faster-rcnn


### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/Flavius1996/VNTS-faster-rcnn.git
    ```

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

3. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

4. Download pre-trained detector from this [link](https://drive.google.com/file/d/1pH9OkaiwzOmHrGBVwBBekmXKyRkqJwUm). This downloaded model need to be placed under the directory
	```Shell
	$FRCN_ROOT/data/VNTSDB/TrainedModel
	```
### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo_vn.py
```
The demo performs detection using a ZF network trained for detection on VNTSDB. Few sample images from [test dataset](https://drive.google.com/open?id=1b3TSfcyODeybJPoVLdwJALcz7JMPPlxx) has been placed under folder
```Shell
cd $FRCN_ROOT/data/demo
```
For the complete testing, [test dataset](https://drive.google.com/open?id=1b3TSfcyODeybJPoVLdwJALcz7JMPPlxx) has to be download and placed in the folder mentioned above

### Beyond the demo: installation for training and testing models

Before starting, you need to download the Vietnam traffic sign datasets from [VietNam Traffic Signs Datasets](https://drive.google.com/open?id=18Wm6viFtG7eScIxRw26Zn3gd5rhsmQ6R). In this implementation, the training and test datasets that were used for the competition ( [training data set](https://drive.google.com/open?id=1XU2jQrHc24KPelBf_C4LxBon1z3uPoUB), [test data set](https://drive.google.com/open?id=1b3TSfcyODeybJPoVLdwJALcz7JMPPlxx) ) is used.

Here, the main goal is to enable Faster R-CNN to detect and classify traffic sign. So, model performance evaluation in test dataset was not carried out. The downloaded test dataset was only used for visual testing. After the dataset is downloaded, prepare the following directory structure. The training zip file contains the following files
- folders
- images (00000.ppm, 00001.ppm...., 00423.ppm)
- gt.txt

Copy all the images into Images directory as shown below. Rename gt.txt as train.txt and keep both gt.txt and train.txt as shown below. 

##### Format Your Dataset
At first, the dataset must be well organzied with the required format.
```
VNTSDB
|-- Annotations
    |-- gt.txt (Annotation files)
|-- Images
    |-- *.ppm (Image files)
|-- ImageSets
    |-- train.txt
```

##### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage
This implementation is tested only for approximate joint training.

To train and test a TSR Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...] [DATASET]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
# --set EXP_DIR seed_rng1701 RNG_SEED 1701
# DATASET to be used for training
```
Example script to train ZF model:
```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end_trainonly.sh 0 ZF vntsdb
```
Trained Fast R-CNN networks are saved under:
```
output/<experiment directory>/<dataset name>/
# Example: output/faster_rcnn_end2end/vntsdb_train
```



# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.


