# Class Activation Map
(In progress) Pytorch implementation of ["CAM: Class Activation Map"](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf). 

## Usage
```
python generate_cam.py
```
By running the code, it will generate class activation maps for randomly sampled imagenet test images. 

## Results
Class Activation Maps from `Resnet18`

|CAM|<img src="https://github.com/minkyu-choi04/ClassActivationMap/blob/master/plots/Selection_109.png" width=200\>|<img src="https://github.com/minkyu-choi04/ClassActivationMap/blob/master/plots/Selection_111.png" width=200\>|<img src="https://github.com/minkyu-choi04/ClassActivationMap/blob/master/plots/Selection_113.png" width=200\>
|--------------------|--------------------|--------------------|--------------------|
|Input image|<img src="https://github.com/minkyu-choi04/ClassActivationMap/blob/master/plots/Selection_110.png" width=200\>|<img src="https://github.com/minkyu-choi04/ClassActivationMap/blob/master/plots/Selection_112.png" width=200\>|<img src="https://github.com/minkyu-choi04/ClassActivationMap/blob/master/plots/Selection_114.png" width=200\>|

## To Do
User interface using `argparse` will be added.
