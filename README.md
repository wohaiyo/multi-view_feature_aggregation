# Multiview Feature Aggregation for Facade Parsing
This project contains the Tensorflow implementation for the proposed method for facade parsing: [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9257006/).

### Introduction
Facade image parsing is essential to the semantic understanding and 3-D reconstruction of urban scenes. Considering the occlusion and appearance ambiguity in single-view
images and the easy acquisition of multiple views, in this letter, we propose a multiview enhanced deep architecture for facade parsing. The highlight of this architecture is a cross-view feature aggregation module that can learn to choose and fuse useful convolutional neural network (CNN) features from nearby views to enhance the representation of a target view. Benefitting from the multiview enhanced representation, the proposed architecture can better deal with the ambiguity and occlusion issues. Moreover, our cross-view feature aggregation module can be straightforwardly integrated into existing single-image parsing frameworks. Extensive comparison experiments and ablation studies are conducted to demonstrate the good performance of the proposed method and the validity and transportability of the cross-view feature aggregation module.
<p align="center"><img width="80%" src="data/architecture.png" /></p>

### Installation
- Env: Python 3.6; TensorFlow 0.5-1.5; CUDA 9.0; cuDNN
- Install some packages
```
pip install opencv-python pillow numpy matplotlib imageio .....
```
- Clone this repository
```
git clone https://github.com/wohaiyo/Pyramid_ALKNet
cd Pyramid_ALKNet
```
- One GPU with 11GB is needed

### Dataset
You need to download the RueMonge2014 dataset, and use the script 'data/prepare_ruemonge_data_homo.py' to pre-process the images, put the files in the `data` folder with following structure.
```
├── RueMonge428_seq3_homo
|    ├── train
|    |    ├── img
|    |    └── label
|    ├── val
|    |    ├── img
|    |    └── label
|    ├── train.txt
|    └── val.txt           
```

### Training

- You can run: `python train_gradient.py -h` to check the detail of optional arguments.
Basically, in the `config.py`, you can set the dataset, train type, epochs and batch size, etc.
```
python train_gradient.py
```
- training on RueMonge2014 train set


### Testing
- After training, the checkpoint will be saved at `saves` folder, you can use `eval_multi.py` to get the result.
```
python eval_multi.py
```

## Data augmentation
   In order to make the model to be capable of leveraging information from multi-views, we synthesize occlusions with randomly block at each view in the training stage. The block area of occlusion is a factor that directly determines the model. We experiment under seven block areas: 0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15.  The block area is fixed at 0.075 in the experiments.
    ![Example](https://github.com/wohaiyo/multi-view_feature_aggregation/blob/master/data/data_aug.png)

## Model training
   To train the proposed model, we adopt the standard cross-entropy loss to evaluate its prediction, i.e., the label map of the target view. The loss is defined as
   
   ![CE](https://github.com/wohaiyo/multi-view_feature_aggregation/blob/master/data/ce_black.png)
   
where M is the total number of category and N is the total number of pixels in the output label map, and c is the index of categories and i is the index of pixels. y c,i and p c,i are the ground truth label and the predicted one of pixel i.
    Since each input view can be treated as the target view, we use multi-view losses to train the overall architecture. The total loss is defined as
    
   ![ALL_CE](https://github.com/wohaiyo/multi-view_feature_aggregation/blob/master/data/all_ce.png)
   
where V is the number of input view.
   For views with synthesized occlusions, we define three extra cross-entropy losses when the view number is three on the weighted feature maps { ˆ X n } n={1,2,3} , to make sure that the feature aggregation happens within visible areas of the target view and complementary parts from the nearby views. We do not calculate the loss of the synthetic occlusion area in one view, but only the loss of the visible part of the area in other views, please see more details in the source code.
   
### Acknowledgments & Contact
Please feel free to contact me if you have any other questions.

E-mail: mawenguang@emails.bjut.edu.cn or mawenguang@126.com
   
### Citation

Please consider citing the [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9257006/) if it's helpful for your research.
```
@article{ma2020multiview,
  title={Multiview Feature Aggregation for Facade Parsing},
  author={Ma, Wenguang and Xu, Shibiao and Ma, Wei and Zha, Hongbin},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2020},
  publisher={IEEE}
}
