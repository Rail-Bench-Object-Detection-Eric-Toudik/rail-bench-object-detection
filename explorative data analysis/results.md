# Explorative Data Analysis

## Insights
- Data quality is very good, there are no missings values or NaNs
- There are no invalid Bounding boxes (BB): all BB coordinates are within the image dimensions
- Most of the samples (test and validation) are in 3840x2160 and 4112x2504 resolution
- Classes are Imbalanced, catenary_pole is the most common (44.1% training, 44.8% in validation),
other classes are less common, train (3.3% training, 3.0% validation) and bicycles (1.3% training, 1.5% validation) are the least common
- BBs have much higher height than width
- BBs are mostly tiny:
  - BB-area relative to resolution quantiles: 0.50->0.04%, 0.75->0.17%, 0.90->0.62%, 0.95->1.38%, 0.99->5.81%
  - BB height 131.2 pixel and width 26.8 pixel
- Train BBs are the biggest (area), signal and signal_pole BBs are the smallest
- Percentage of BBs that are close (up to 5 pixels) to the borders: left->1.51%, top->7.47%, right->1.31%, bottom->0.32%
- Most of the BBs are not occluded (occluded-level: 0), some are partially occluded (occluded-level: 1 to 3) and very few are heavily occluded (occluded-level: 4)
- Mostly BBs of the category person and road_vehicle are occluded
- Occluded BBs does not mean that the images are foggy
- Images can be foggy or blurry, and are taken in different weather conditions



## Questions:
- Research how small objects can be in order to be detected by the model (e.g. 10% of image size) and how can we handle for object detection