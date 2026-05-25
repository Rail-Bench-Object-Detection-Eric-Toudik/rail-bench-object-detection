# Description
- All models are trained on the same training dataset.
- The metrics are calculated on the validation set using the RailBench Toolkit.
- For all models the same splits where used for training and validation.
# baseline_yolo11s_300epochs_no_augmentation_1024_eric
## Config:
## Training Configuration

## Training Configuration

| Category | Parameter | Value |
|---|---|---:|
| Training | epochs | 300 |
| Training | batch_size | 12 |
| Training | image_size | 1024 |
| Training | model_name | `yolo11s` |
| Training | seed | 42 |
| Learning Rate | lr0 | 0.01 |
| Learning Rate | lrf | 0.01 |
| Augmentation | auto_augment | `randaugment` |
| Color Augmentation | hsv_h | 0.01 |
| Color Augmentation | hsv_s | 0.5 |
| Color Augmentation | hsv_v | 0.3 |
| Geometric Augmentation | translate | 0.1 |
| Geometric Augmentation | scale | 0.4 |
| Geometric Augmentation | fliplr | 0.5 |

## Metrics


| Metric     |   Value |
| ---------- | -----: |
| AP         | 0.1460 |
| AP50       | 0.2852 |
| AP75       | 0.1293 |
| APeasy     | 0.2930 |
| APmoderate | 0.2127 |
| APhard     | 0.1460 |

| Recall     |   Value |
| ---------- | -----: |
| AReasy     | 0.3754 |
| ARmoderate | 0.2738 |
| ARhard     | 0.1881 |
| AR1        | 0.0730 |
| AR10       | 0.1690 |
| AR100      | 0.1881 |
| AR200      | 0.1881 |

|  IoU | catenary_pole | road_vehicle | signal |  train | signal_pole | person | bicycle |
| ---: | ------------: | -----------: | -----: | -----: | ----------: | -----: | ------: |
| 0.50 |        0.4750 |       0.2788 | 0.3152 | 0.3345 |      0.1647 | 0.4171 |  0.0112 |
| 0.55 |        0.4524 |       0.2355 | 0.2968 | 0.3149 |      0.1364 | 0.3871 |  0.0112 |
| 0.60 |        0.4191 |       0.1934 | 0.2850 | 0.2885 |      0.1054 | 0.3474 |  0.0112 |
| 0.65 |        0.3735 |       0.1635 | 0.2556 | 0.2625 |      0.0828 | 0.2951 |  0.0103 |
| 0.70 |        0.3197 |       0.1228 | 0.2229 | 0.2216 |      0.0564 | 0.2392 |  0.0075 |
| 0.75 |        0.2465 |       0.0866 | 0.1884 | 0.1797 |      0.0332 | 0.1662 |  0.0042 |
| 0.80 |        0.1600 |       0.0649 | 0.1473 | 0.1350 |      0.0151 | 0.1081 |  0.0004 |
| 0.85 |        0.0852 |       0.0480 | 0.0945 | 0.0877 |      0.0075 | 0.0497 |  0.0004 |
| 0.90 |        0.0265 |       0.0240 | 0.0385 | 0.0452 |      0.0025 | 0.0184 |  0.0004 |
| 0.95 |        0.0040 |       0.0104 | 0.0107 | 0.0102 |      0.0004 | 0.0073 |  0.0004 |

# baseline_yolo11s_300epochs_blur_augmentation_1024_eric
## Config

## Training Configuration

| Category | Parameter | Value |
|---|---|---:|
| Training | epochs | 300 |
| Training | batch_size | 12 |
| Training | image_size | 1024 |
| Training | model_name | `yolo11s` |
| Training | seed | 42 |
| Learning Rate | lr0 | 0.01 |
| Learning Rate | lrf | 0.01 |
| Augmentation | auto_augment | `randaugment` |
| Color Augmentation | hsv_h | 0.01 |
| Color Augmentation | hsv_s | 0.5 |
| Color Augmentation | hsv_v | 0.3 |
| Geometric Augmentation | translate | 0.1 |
| Geometric Augmentation | scale | 0.4 |
| Geometric Augmentation | fliplr | 0.5 |
| Blur Augmentation | blur | `True` |
| Blur Augmentation | blur_limit | 3 |
| Blur Augmentation | blur_p | 0.05 |
| Motion Blur Augmentation | motion_blur | `True` |
| Motion Blur Augmentation | motion_blur_limit | 3 |
| Motion Blur Augmentation | motion_blur_p | 0.05 |

## Metrics
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1696 |
| AP50 | 0.3224 |
| AP75 | 0.1574 |
| APeasy | 0.3326 |
| APmoderate | 0.2445 |
| APhard | 0.1696 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4131 |
| ARmoderate | 0.3067 |
| ARhard | 0.2131 |
| AR1 | 0.0825 |
| AR10 | 0.1920 |
| AR100 | 0.2131 |
| AR200 | 0.2131 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | train | road_vehicle | signal | person | signal_pole | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.4997 | 0.4171 | 0.3063 | 0.3348 | 0.4641 | 0.1855 | 0.0494 |
| 0.55 | 0.4781 | 0.3676 | 0.2718 | 0.3167 | 0.4363 | 0.1617 | 0.0434 |
| 0.60 | 0.4475 | 0.3362 | 0.2357 | 0.2905 | 0.3939 | 0.1308 | 0.0391 |
| 0.65 | 0.4091 | 0.3049 | 0.1955 | 0.2639 | 0.3373 | 0.1041 | 0.0366 |
| 0.70 | 0.3562 | 0.2774 | 0.1617 | 0.2357 | 0.2648 | 0.0801 | 0.0318 |
| 0.75 | 0.2869 | 0.2273 | 0.1248 | 0.1966 | 0.1972 | 0.0476 | 0.0212 |
| 0.80 | 0.2040 | 0.1680 | 0.0844 | 0.1600 | 0.1244 | 0.0225 | 0.0184 |
| 0.85 | 0.1134 | 0.1123 | 0.0607 | 0.1087 | 0.0613 | 0.0095 | 0.0127 |
| 0.90 | 0.0408 | 0.0680 | 0.0249 | 0.0463 | 0.0174 | 0.0058 | 0.0005 |
| 0.95 | 0.0109 | 0.0104 | 0.0031 | 0.0112 | 0.0058 | 0.0002 | 0.0005 |

# baseline_yolo11s_300epochs_basic_augmentation_1024_toudik
## Training Configuration

| Category | Parameter | Value |
|---|---|---:|
| Training | epochs | 300 |
| Training | batch_size | 12 |
| Training | image_size | 1024 |
| Training | model_name | `yolo11s` |
| Training | seed | 42 |
| Learning Rate | lr0 | 0.01 |
| Learning Rate | lrf | 0.01 |
| Augmentation | auto_augment | `randaugment` |
| Color Augmentation | hsv_h | 0.01 |
| Color Augmentation | hsv_s | 0.5 |
| Color Augmentation | hsv_v | 0.3 |
| Geometric Augmentation | translate | 0.1 |
| Geometric Augmentation | scale | 0.4 |
| Geometric Augmentation | fliplr | 0.5 |

## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1728 |
| AP50 | 0.3283 |
| AP75 | 0.1549 |
| APeasy | 0.3365 |
| APmoderate | 0.2469 |
| APhard | 0.1728 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4176 |
| ARmoderate | 0.3082 |
| ARhard | 0.2153 |
| AR1 | 0.0844 |
| AR10 | 0.1929 |
| AR100 | 0.2153 |
| AR200 | 0.2153 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | train | signal | signal_pole | person | road_vehicle | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5064 | 0.4024 | 0.3436 | 0.1955 | 0.4592 | 0.3171 | 0.0738 |
| 0.55 | 0.4846 | 0.3909 | 0.3125 | 0.1739 | 0.4347 | 0.2904 | 0.0683 |
| 0.60 | 0.4506 | 0.3375 | 0.2920 | 0.1399 | 0.3965 | 0.2599 | 0.0683 |
| 0.65 | 0.4043 | 0.3133 | 0.2654 | 0.0979 | 0.3506 | 0.2124 | 0.0659 |
| 0.70 | 0.3493 | 0.2717 | 0.2353 | 0.0704 | 0.2663 | 0.1700 | 0.0391 |
| 0.75 | 0.2835 | 0.2344 | 0.2020 | 0.0460 | 0.1893 | 0.1083 | 0.0212 |
| 0.80 | 0.2056 | 0.1803 | 0.1610 | 0.0205 | 0.1264 | 0.0690 | 0.0160 |
| 0.85 | 0.1118 | 0.1296 | 0.1172 | 0.0094 | 0.0639 | 0.0444 | 0.0160 |
| 0.90 | 0.0431 | 0.0521 | 0.0488 | 0.0020 | 0.0253 | 0.0256 | 0.0045 |
| 0.95 | 0.0042 | 0.0024 | 0.0062 | 0.0001 | 0.0109 | 0.0037 | 0.0045 |

# baseline_yolo11s_300epochs_blur_augmentation_1024_toudik
## Training Configuration

| Category | Parameter | Value |
|---|---|---:|
| Training | epochs | 300 |
| Training | batch_size | 12 |
| Training | image_size | 1024 |
| Training | model_name | `yolo11s` |
| Training | seed | 42 |
| Learning Rate | lr0 | 0.01 |
| Learning Rate | lrf | 0.01 |
| Augmentation | auto_augment | `randaugment` |
| Color Augmentation | hsv_h | 0.01 |
| Color Augmentation | hsv_s | 0.5 |
| Color Augmentation | hsv_v | 0.3 |
| Geometric Augmentation | translate | 0.1 |
| Geometric Augmentation | scale | 0.4 |
| Geometric Augmentation | fliplr | 0.5 |
| Blur Augmentation | blur | `True` |
| Blur Augmentation | blur_limit | `3,10` |
| Blur Augmentation | blur_p | 0.05 |
| Motion Blur Augmentation | motion_blur | `True` |
| Motion Blur Augmentation | motion_blur_limit | `3,10` |
| Motion Blur Augmentation | motion_blur_p | 0.05 |
## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1698 |
| AP50 | 0.3244 |
| AP75 | 0.1571 |
| APeasy | 0.3319 |
| APmoderate | 0.2444 |
| APhard | 0.1698 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4070 |
| ARmoderate | 0.3020 |
| ARhard | 0.2094 |
| AR1 | 0.0835 |
| AR10 | 0.1900 |
| AR100 | 0.2094 |
| AR200 | 0.2094 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | train | person | signal_pole | road_vehicle | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5105 | 0.3287 | 0.4496 | 0.4520 | 0.2077 | 0.2828 | 0.0395 |
| 0.55 | 0.4893 | 0.3086 | 0.4078 | 0.4156 | 0.1687 | 0.2539 | 0.0395 |
| 0.60 | 0.4565 | 0.2967 | 0.3543 | 0.3790 | 0.1381 | 0.2231 | 0.0287 |
| 0.65 | 0.4119 | 0.2649 | 0.3051 | 0.3316 | 0.1181 | 0.1940 | 0.0234 |
| 0.70 | 0.3589 | 0.2297 | 0.2679 | 0.2668 | 0.0838 | 0.1600 | 0.0208 |
| 0.75 | 0.2876 | 0.1951 | 0.2157 | 0.1970 | 0.0540 | 0.1301 | 0.0200 |
| 0.80 | 0.1991 | 0.1507 | 0.1687 | 0.1321 | 0.0284 | 0.0879 | 0.0200 |
| 0.85 | 0.1126 | 0.0989 | 0.1083 | 0.0642 | 0.0128 | 0.0578 | 0.0200 |
| 0.90 | 0.0434 | 0.0505 | 0.0465 | 0.0241 | 0.0069 | 0.0325 | 0.0147 |
| 0.95 | 0.0061 | 0.0081 | 0.0015 | 0.0110 | 0.0009 | 0.0055 | 0.0058 |


# SAHI on Inference only with  baseline_yolo11s_300epochs_basic_augmentation_1024_toudik


| Category               | Parameter   |         Value |
|------------------------|-------------|--------------:|
| Training               | epochs      |           300 |
| Training               | batch_size  |            12 |
| Training               | image_size  |          1024 |
| Training               | model_name  |     `yolo11s` |
| Training               | seed        |            42 |
| Learning Rate          | lr0         |          0.01 |
| Learning Rate          | lrf         |          0.01 |
| Augmentation           | auto_augment | `randaugment` |
| Color Augmentation     | hsv_h       |          0.01 |
| Color Augmentation     | hsv_s       |           0.5 |
| Color Augmentation     | hsv_v       |           0.3 |
| Geometric Augmentation | translate   |           0.1 |
| Geometric Augmentation | scale       |           0.4 |
| Geometric Augmentation | fliplr      |           0.5 |
| SAHI                   | Slice_Height|          1024 |
| SAHI                   | SLICE_WIDTH |          1024 |
| SAHI                   | OVERLAP_HEIGHT_RATIO|           0.2 |
| SAHI                   | OVERLAP_WIDTH_RATIO|           0.2 |

