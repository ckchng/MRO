# MRO

Author: Chee-Kheng Chng

Given monocular images, MRO estimates their absolute orientations. 
It is 
1) robust against pure rotation motion (equipped with the relative rotation estimator from openGV (https://laurentkneip.github.io/opengv/)) 
2) capable of loop closing
3) a constant time rotation avergaing solver (due to its incremental nature).

It is an extension to IRotAvg from https://github.com/ajparra/iRotAvg.

## License

MRO is released under a GPLv3 license. 

For a closed-source version of MRO (e.g., for commercial purposes), please contact the author.

For an academic use of MRO, please cite
[C.-K. Chng, Á. Parra, T.-J. Chin, Y. Latif: Monocular Rotational Odometry with Incremental Rotation Averaging and Loop Closure, DICTA 2020](https://arxiv.org/pdf/2010.01872.pdf)


## Dependencies
Tested on:
- Eigen 3.3.4
sudo apt install libeigen3-dev

- SuiteSparse
sudo apt-get install libsuitesparse-dev

- opencv 4.0.0
https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/ provides a good guide.

sudo apt-get install libboost-all-dev
- Boost (Filesystem) 1.6.5

- opengv 1.0.0 (refer to the [official site](https://laurentkneip.github.io/opengv/) for potential installation FAQs)
cd third_party/opengv
mkdir build
cd build
cmake ..
make 
cmake -DCMAKE_INSTALL_PREFIX=/path/to/local_install -P cmake_install.cmake 

## Compilation
Finally, to install MRO

*first edit line 8 and 10 in /src/CMakelist.txt to appropriate directories*

Then,

cd /dir/to/MRO
mkdir build
cd build
cmake ..
make 

(binary is compiled inside src)

## Demo
The demo.sh bash file in written to run KITTI seq00. The sequence (and the entire dataset) can be downloaded [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). 

Edit the I/O paths appropriately. 

ORB feature extractor used in MRO takes *config.yaml* and *ORBvoc.txt* as inputs, both are provided in /data. The current config is tuned according to the camera parameters of KITTI sequence00. It needs to be adjusted accordingly when running on other datasets.

Run ./demo.sh 

## Hyperparameters
As seen in demo.sh

- MIN_SET : minimal set to run ransac for opengv's relative rotation estimator, [original paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Kneip_Direct_Optimization_of_2013_ICCV_paper.pdf) proposes 10.

- winsize : the number of neighbouring window of frames that MRO performs feature matching on.

- img_width: image width

- img_height: image height

In /src/IRotAvg.cpp
- rotavg_win_size : number of nodes that the underlying incremental rotation averaging solver takes in during optimisation

- vg_min_matches : minimum number of feature matches to be considered a pair of frames that are viewing the same scene.



