# Text2Mesh
This repository contains code for the CVPR submission #10348 "There and Back Again: 3D Mesh Sequence Generation from Text Using Back-Translation". 

## Installation
Before using this code, please install the dependencies inside environment.yml.

Additionally, you will have to install the dependencies and download the models for SMPLify-X:
https://github.com/vchoutas/smplify-x

## Running the Demo
- unip the model weights in demo/eval_model/sign_sample_model_200_r0t1_all_256_1024_2D_pose_hands
- unzip the test and dev archive in demo/TextData and put the files in the TextData folder
- unzip the test archive under demo/Data
- call run_demo.py with the following arguments:
--config
/path/to/Text2Mesh/Pose2Mesh/smplifyx/cfg_files/fit_smplx.yaml
--data_folder
/path/to/Text2Mesh/demo/Data
--output_folder
/path/to/Text2Mesh/demo/Data
--visualize=False
--model_folder
/path/to/Text2Mesh/Pose2Mesh/smplifyx/smplx/models/smplx
--vposer_ckpt
/path/to/Text2Mesh/Pose2Mesh/smplifyxs/smplifyx/V02_05
--part_segm_fn
smplx_parts_segm.pkl

Full evaluation and training code will be added upon acceptance of the manuscript.

