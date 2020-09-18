# multi-view_feature_aggregation
This repository is built for our paper Multi-view Feature Aggregation for Facade Parsing. The source code and more details will be released in the near future.

## Data augmentation
   In order to make the model to be capable of leveraging information from multi-views, we synthesize occlusions with randomly block at each view in the training stage. The block area of occlusion is a factor that directly determines the model. We experiment under seven block areas: 0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15.  The block area is fixed at 0.075 in the experiments.
   

## Model training
