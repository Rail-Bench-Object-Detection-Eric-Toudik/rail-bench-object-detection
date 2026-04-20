# Explorative Data Analysis

Really necessary?
- Distribution of annotations per image necessary
- Bounding Box sizes in relation to image size
- CDF of Bounding box sizes

Questions:
- Objects close to border -> why mean?
- Research how small objects can be in order to be detected by the model (e.g. 10% of image size) and how can we handle for object detection

Improvements:
- Only use Bounding Box sizes in relation to image size categorized by class (person, car, bicycle...)
- move notebook to explorative_data_analysis folder