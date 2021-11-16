# Text2Mesh
This repository contains code for the CVPR submission #10348 "There and Back Again: 3D Mesh Sequence Generation from Text Using Back-Translation". 

## Installation
Before using this code, please install the dependencies inside environment.yml.

Additionally, you will have to install the dependencies and download the models for SMPLify-X:
https://github.com/vchoutas/smplify-x

## If you are a Reviewer
We understand you might not have time to follow the next steps. In this case, we have created a demo for you:



## Data
The PHOENIX14T Dataset is available at :https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/. You will have to run OpenPose to extract 2D keypoint information: https://github.com/CMU-Perceptual-Computing-Lab/openpose. We accumulated the .json files provided by openpose in matlab objects for train, test, and validation data, respectively. You can then use the script preprocessing/pre-processing.py to generate data for Text2Pose. If you want to make use of the back-translation loss you will also have to run make_data_for_slt.py afterwards. Please make sure to edit paths inside the scripts accordingly. 

## Training
To train the back-translation model:
To train the Text2Pose model:

## Inference

