from ultralytics.data.converter import convert_coco
path = "../data/coco_annotations/objects"
convert_coco(labels_dir=path)
