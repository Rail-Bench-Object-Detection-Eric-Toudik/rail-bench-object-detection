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
| Augmentation | auto_augment | `None` |

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


# SAHI on Inference only with  baseline_yolo11s_300epochs_basic_augmentation_1024_toudik (768x1536)


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
| SAHI                   | Slice_Height|          1536 |
| SAHI                   | SLICE_WIDTH |           768 |
| SAHI                   | OVERLAP_HEIGHT_RATIO|           0.2 |
| SAHI                   | OVERLAP_WIDTH_RATIO|           0.3 |

## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1528 |
| AP50 | 0.3129 |
| AP75 | 0.1283 |
| APeasy | 0.2046 |
| APmoderate | 0.1831 |
| APhard | 0.1528 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.3290 |
| ARmoderate | 0.2945 |
| ARhard | 0.2533 |
| AR1 | 0.0764 |
| AR10 | 0.2107 |
| AR100 | 0.2533 |
| AR200 | 0.2533 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | train | signal_pole | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.4503 | 0.4106 | 0.2709 | 0.1887 | 0.2602 | 0.4826 | 0.1270 |
| 0.55 | 0.4032 | 0.3851 | 0.2263 | 0.1498 | 0.2351 | 0.4547 | 0.1115 |
| 0.60 | 0.3464 | 0.3606 | 0.1913 | 0.1005 | 0.2095 | 0.4180 | 0.1030 |
| 0.65 | 0.2687 | 0.3266 | 0.1538 | 0.0607 | 0.1849 | 0.3680 | 0.1009 |
| 0.70 | 0.1943 | 0.2907 | 0.1293 | 0.0314 | 0.1493 | 0.3116 | 0.0712 |
| 0.75 | 0.1260 | 0.2466 | 0.1080 | 0.0180 | 0.1101 | 0.2357 | 0.0538 |
| 0.80 | 0.0652 | 0.1976 | 0.0891 | 0.0092 | 0.0929 | 0.1620 | 0.0476 |
| 0.85 | 0.0245 | 0.1294 | 0.0715 | 0.0060 | 0.0612 | 0.0943 | 0.0329 |
| 0.90 | 0.0044 | 0.0534 | 0.0355 | 0.0025 | 0.0249 | 0.0398 | 0.0076 |
| 0.95 | 0.0003 | 0.0037 | 0.0039 | 0.0001 | 0.0052 | 0.0061 | 0.0034 |



# SAHI on Inference only with  baseline_yolo11s_300epochs_basic_augmentation_1024_toudik (1024x1024)


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

## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1870 |
| AP50 | 0.3774 |
| AP75 | 0.1661 |
| APeasy | 0.2842 |
| APmoderate | 0.2426 |
| APhard | 0.1870 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4078 |
| ARmoderate | 0.3456 |
| ARhard | 0.2720 |
| AR1 | 0.0831 |
| AR10 | 0.2339 |
| AR100 | 0.2720 |
| AR200 | 0.2720 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | train | signal | signal_pole | bicycle | road_vehicle | person |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6380 | 0.2109 | 0.4744 | 0.2824 | 0.1524 | 0.3259 | 0.5577 |
| 0.55 | 0.5905 | 0.1802 | 0.4568 | 0.2079 | 0.1237 | 0.2993 | 0.5167 |
| 0.60 | 0.5258 | 0.1448 | 0.4181 | 0.1590 | 0.0996 | 0.2563 | 0.4790 |
| 0.65 | 0.4563 | 0.1380 | 0.3808 | 0.1106 | 0.0950 | 0.2088 | 0.4170 |
| 0.70 | 0.3731 | 0.1086 | 0.3417 | 0.0629 | 0.0767 | 0.1621 | 0.3441 |
| 0.75 | 0.2781 | 0.0891 | 0.2977 | 0.0361 | 0.0579 | 0.1249 | 0.2793 |
| 0.80 | 0.1709 | 0.0664 | 0.2439 | 0.0212 | 0.0372 | 0.0915 | 0.1894 |
| 0.85 | 0.0811 | 0.0484 | 0.1697 | 0.0076 | 0.0111 | 0.0627 | 0.1021 |
| 0.90 | 0.0246 | 0.0252 | 0.0798 | 0.0021 | 0.0062 | 0.0365 | 0.0381 |
| 0.95 | 0.0018 | 0.0014 | 0.0082 | 0.0003 | 0.0058 | 0.0084 | 0.0106 |


# SAHI on Inference only with  baseline_yolo11s_300epochs_basic_augmentation_1024_toudik (768x1536)


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
| SAHI                   | Slice_Height|          2048 |
| SAHI                   | SLICE_WIDTH |           768 |
| SAHI                   | OVERLAP_HEIGHT_RATIO|           0.2 |
| SAHI                   | OVERLAP_WIDTH_RATIO|           0.3 |

## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1941 |
| AP50 | 0.3775 |
| AP75 | 0.1752 |
| APeasy | 0.3198 |
| APmoderate | 0.2633 |
| APhard | 0.1941 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4436 |
| ARmoderate | 0.3656 |
| ARhard | 0.2747 |
| AR1 | 0.0867 |
| AR10 | 0.2397 |
| AR100 | 0.2747 |
| AR200 | 0.2747 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | train | signal_pole | person | road_vehicle | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6439 | 0.4814 | 0.2270 | 0.2794 | 0.5498 | 0.3383 | 0.1225 |
| 0.55 | 0.6156 | 0.4490 | 0.1856 | 0.2198 | 0.5120 | 0.3103 | 0.1176 |
| 0.60 | 0.5733 | 0.4173 | 0.1501 | 0.1691 | 0.4702 | 0.2675 | 0.1029 |
| 0.65 | 0.5228 | 0.3781 | 0.1320 | 0.1245 | 0.4204 | 0.2239 | 0.0970 |
| 0.70 | 0.4495 | 0.3497 | 0.1104 | 0.0815 | 0.3421 | 0.1705 | 0.0497 |
| 0.75 | 0.3631 | 0.3027 | 0.0905 | 0.0521 | 0.2498 | 0.1343 | 0.0337 |
| 0.80 | 0.2496 | 0.2519 | 0.0658 | 0.0272 | 0.1810 | 0.1017 | 0.0278 |
| 0.85 | 0.1281 | 0.1731 | 0.0368 | 0.0142 | 0.1097 | 0.0653 | 0.0145 |
| 0.90 | 0.0359 | 0.0675 | 0.0138 | 0.0051 | 0.0423 | 0.0389 | 0.0066 |
| 0.95 | 0.0032 | 0.0102 | 0.0011 | 0.0050 | 0.0128 | 0.0114 | 0.0062 |



# Trained on image slices, inference non-sliced (baseline_yolo11s_60epochs_SAHI_Training_1024_eric)

## Training Configuration

| Category | Parameter |         Value |
|---|---|--------------:|
| Training | epochs |            60 |
| Training | batch_size |            12 |
| Training | image_size |          1024 |
| Training | model_name |     `yolo11s` |
| Training | seed |            42 |
| Training | workers |             2 |
| Learning Rate | lr0 |          0.01 |
| Learning Rate | lrf |          0.01 |
| Augmentation | auto_augment | `randaugment` |
| Color Augmentation | hsv_h |          0.01 |
| Color Augmentation | hsv_s |           0.5 |
| Color Augmentation | hsv_v |           0.3 |
| Geometric Augmentation | translate |           0.1 |
| Geometric Augmentation | scale |           0.4 |
| Geometric Augmentation | fliplr |           0.5 |
| Blur Augmentation | blur |       `False` |
| Blur Augmentation | blur_limit |        `3,10` |
| Blur Augmentation | blur_p |          0.05 |
| Motion Blur Augmentation | motion_blur |       `False` |
| Motion Blur Augmentation | motion_blur_limit |        `3,10` |
| Motion Blur Augmentation | motion_blur_p |          0.05 |
| SAHI | use_sahi |        `True` |
| SAHI | slice_height |          2048 |
| SAHI | slice_width |           768 |
| SAHI | overlap_height_ratio |           0.2 |
| SAHI | overlap_width_ratio |           0.3 |

## Evaluation Results: Direct Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1528 |
| AP50 | 0.2836 |
| AP75 | 0.1453 |
| APeasy | 0.3189 |
| APmoderate | 0.2263 |
| APhard | 0.1528 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.3948 |
| ARmoderate | 0.2834 |
| ARhard | 0.1912 |
| AR1 | 0.0752 |
| AR10 | 0.1774 |
| AR100 | 0.1912 |
| AR200 | 0.1912 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | road_vehicle | signal_pole | person | train | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.4763 | 0.3090 | 0.2861 | 0.1767 | 0.3757 | 0.3192 | 0.0424 |
| 0.55 | 0.4557 | 0.2913 | 0.2617 | 0.1531 | 0.3537 | 0.2925 | 0.0339 |
| 0.60 | 0.4273 | 0.2802 | 0.2306 | 0.1257 | 0.3262 | 0.2744 | 0.0280 |
| 0.65 | 0.3926 | 0.2539 | 0.1952 | 0.1021 | 0.2856 | 0.2368 | 0.0195 |
| 0.70 | 0.3479 | 0.2312 | 0.1627 | 0.0739 | 0.2211 | 0.2166 | 0.0149 |
| 0.75 | 0.2861 | 0.1972 | 0.1247 | 0.0486 | 0.1692 | 0.1762 | 0.0149 |
| 0.80 | 0.2081 | 0.1601 | 0.0945 | 0.0282 | 0.1119 | 0.1303 | 0.0149 |
| 0.85 | 0.1184 | 0.1174 | 0.0598 | 0.0176 | 0.0595 | 0.0645 | 0.0012 |
| 0.90 | 0.0434 | 0.0439 | 0.0300 | 0.0031 | 0.0236 | 0.0342 | 0.0012 |
| 0.95 | 0.0078 | 0.0094 | 0.0109 | 0.0025 | 0.0035 | 0.0069 | 0.0012 |



#  Trained on image slices, inference sliced (baseline_yolo11s_60epochs_SAHI_Training_1024_eric)

## Training Configuration

| Category | Parameter                               |         Value |
|---|-----------------------------------------|--------------:|
| Training | epochs                                  |            60 |
| Training | batch_size                              |            12 |
| Training | image_size                              |          1024 |
| Training | model_name                              |     `yolo11s` |
| Training | seed                                    |            42 |
| Training | workers                                 |             2 |
| Learning Rate | lr0                                     |          0.01 |
| Learning Rate | lrf                                     |          0.01 |
| Augmentation | auto_augment                            | `randaugment` |
| Color Augmentation | hsv_h                                   |          0.01 |
| Color Augmentation | hsv_s                                   |           0.5 |
| Color Augmentation | hsv_v                                   |           0.3 |
| Geometric Augmentation | translate                               |           0.1 |
| Geometric Augmentation | scale                                   |           0.4 |
| Geometric Augmentation | fliplr                                  |           0.5 |
| Blur Augmentation | blur                                    |       `False` |
| Blur Augmentation | blur_limit                              |        `3,10` |
| Blur Augmentation | blur_p                                  |          0.05 |
| Motion Blur Augmentation | motion_blur                             |       `False` |
| Motion Blur Augmentation | motion_blur_limit                       |        `3,10` |
| Motion Blur Augmentation | motion_blur_p                           |          0.05 |
| SAHI | use_sahi                                |        `True` |
| SAHI | slice_height (train,inference)          |          2048 |
| SAHI | slice_width  (train,inference)          |           768 |
| SAHI | overlap_height_ratio (train, inference) |           0.2 |
| SAHI | overlap_width_ratio  (train, inference)          |           0.3 |

## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2128 |
| AP50 | 0.3985 |
| AP75 | 0.1997 |
| APeasy | 0.3664 |
| APmoderate | 0.2979 |
| APhard | 0.2128 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4898 |
| ARmoderate | 0.3972 |
| ARhard | 0.2903 |
| AR1 | 0.0924 |
| AR10 | 0.2553 |
| AR100 | 0.2903 |
| AR200 | 0.2903 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | road_vehicle | signal_pole | train | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6635 | 0.5039 | 0.3588 | 0.3192 | 0.2108 | 0.5563 | 0.1767 |
| 0.55 | 0.6314 | 0.4763 | 0.3222 | 0.2676 | 0.1794 | 0.5285 | 0.1656 |
| 0.60 | 0.5971 | 0.4519 | 0.2803 | 0.2214 | 0.1434 | 0.4927 | 0.1415 |
| 0.65 | 0.5527 | 0.4230 | 0.2460 | 0.1759 | 0.1237 | 0.4384 | 0.1162 |
| 0.70 | 0.4903 | 0.3873 | 0.1912 | 0.1162 | 0.0935 | 0.3722 | 0.0827 |
| 0.75 | 0.4051 | 0.3415 | 0.1521 | 0.0812 | 0.0678 | 0.2767 | 0.0736 |
| 0.80 | 0.2851 | 0.2861 | 0.1126 | 0.0518 | 0.0610 | 0.1797 | 0.0548 |
| 0.85 | 0.1632 | 0.2016 | 0.0661 | 0.0236 | 0.0396 | 0.1040 | 0.0368 |
| 0.90 | 0.0524 | 0.1039 | 0.0308 | 0.0042 | 0.0220 | 0.0461 | 0.0216 |
| 0.95 | 0.0115 | 0.0195 | 0.0088 | 0.0017 | 0.0003 | 0.0068 | 0.0064 |



#  Oversampling (baseline_yolo11s_300epochs_basic_augmentation_1024_toudik)

## Training Configuration

| Category               | Parameter |         Value |
|------------------------|---|--------------:|
| Training               | epochs |           300 |
| Training               | batch_size |            12 |
| Training               | image_size |          1024 |
| Training               | model_name |     `yolo11s` |
| Training               | seed |            42 |
| Learning Rate          | lr0 |          0.01 |
| Learning Rate          | lrf |          0.01 |
| Augmentation           | auto_augment | `randaugment` |
| Color Augmentation     | hsv_h |          0.01 |
| Color Augmentation     | hsv_s |           0.5 |
| Color Augmentation     | hsv_v |           0.3 |
| Geometric Augmentation | translate |           0.1 |
| Geometric Augmentation | scale |           0.4 |
| Geometric Augmentation | fliplr |           0.5 |
| Oversampling           | ROAD_VEHICLE_FACTOR |             2 |
| Oversampling  | BICYCLE_FACTOR |           4 |


## Evaluation Results: Direct Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1608 |
| AP50 | 0.3051 |
| AP75 | 0.1473 |
| APeasy | 0.3164 |
| APmoderate | 0.2319 |
| APhard | 0.1608 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.3869 |
| ARmoderate | 0.2857 |
| ARhard | 0.1974 |
| AR1 | 0.0774 |
| AR10 | 0.1804 |
| AR100 | 0.1974 |
| AR200 | 0.1974 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | road_vehicle | person | train | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5123 | 0.3270 | 0.1943 | 0.2774 | 0.4356 | 0.3712 | 0.0180 |
| 0.55 | 0.4904 | 0.3098 | 0.1582 | 0.2516 | 0.4035 | 0.3376 | 0.0161 |
| 0.60 | 0.4571 | 0.2740 | 0.1277 | 0.2209 | 0.3741 | 0.3108 | 0.0161 |
| 0.65 | 0.4134 | 0.2516 | 0.0973 | 0.1959 | 0.3236 | 0.2885 | 0.0092 |
| 0.70 | 0.3478 | 0.2121 | 0.0687 | 0.1593 | 0.2632 | 0.2528 | 0.0027 |
| 0.75 | 0.2787 | 0.1817 | 0.0427 | 0.1126 | 0.1910 | 0.2216 | 0.0027 |
| 0.80 | 0.1926 | 0.1500 | 0.0254 | 0.0768 | 0.1095 | 0.1807 | 0.0027 |
| 0.85 | 0.1113 | 0.1040 | 0.0131 | 0.0533 | 0.0586 | 0.1316 | 0.0027 |
| 0.90 | 0.0431 | 0.0495 | 0.0099 | 0.0234 | 0.0208 | 0.0560 | 0.0005 |
| 0.95 | 0.0112 | 0.0069 | 0.0000 | 0.0107 | 0.0077 | 0.0030 | 0.0005 |



#  Yolov11M (baseline_yolo11s_300epochs_basic_augmentation_1024_toudik)

## Training Configuration

| Category               | Parameter |         Value |
|------------------------|---|--------------:|
| Training               | epochs |           300 |
| Training               | batch_size |            12 |
| Training               | image_size |          1024 |
| Training               | model_name |     `yolo11m` |
| Training               | seed |            42 |
| Learning Rate          | lr0 |          0.01 |
| Learning Rate          | lrf |          0.01 |
| Augmentation           | auto_augment | `randaugment` |
| Color Augmentation     | hsv_h |          0.01 |
| Color Augmentation     | hsv_s |           0.5 |
| Color Augmentation     | hsv_v |           0.3 |
| Geometric Augmentation | translate |           0.1 |
| Geometric Augmentation | scale |           0.4 |
| Geometric Augmentation | fliplr |           0.5 |


## Evaluation Results: Direct Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2040 |
| AP50 | 0.3730 |
| AP75 | 0.1925 |
| APeasy | 0.3839 |
| APmoderate | 0.2914 |
| APhard | 0.2040 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4707 |
| ARmoderate | 0.3604 |
| ARhard | 0.2534 |
| AR1 | 0.0949 |
| AR10 | 0.2285 |
| AR100 | 0.2534 |
| AR200 | 0.2534 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal_pole | signal | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5522 | 0.2336 | 0.4066 | 0.4225 | 0.3687 | 0.5183 | 0.1089 |
| 0.55 | 0.5305 | 0.1990 | 0.3805 | 0.3868 | 0.3533 | 0.4940 | 0.1086 |
| 0.60 | 0.5006 | 0.1668 | 0.3544 | 0.3605 | 0.3079 | 0.4516 | 0.0861 |
| 0.65 | 0.4660 | 0.1327 | 0.3257 | 0.3348 | 0.2708 | 0.3850 | 0.0639 |
| 0.70 | 0.4081 | 0.0971 | 0.2872 | 0.2995 | 0.2216 | 0.3124 | 0.0570 |
| 0.75 | 0.3377 | 0.0612 | 0.2421 | 0.2693 | 0.1618 | 0.2395 | 0.0360 |
| 0.80 | 0.2431 | 0.0361 | 0.1946 | 0.2293 | 0.1121 | 0.1512 | 0.0100 |
| 0.85 | 0.1436 | 0.0211 | 0.1433 | 0.1513 | 0.0733 | 0.0824 | 0.0060 |
| 0.90 | 0.0584 | 0.0119 | 0.0604 | 0.0959 | 0.0383 | 0.0355 | 0.0041 |
| 0.95 | 0.0075 | 0.0017 | 0.0151 | 0.0233 | 0.0124 | 0.0118 | 0.0041 |



#  Yolov11M with SAHI - inference only -  (baseline_yolo11s_300epochs_basic_augmentation_1024_toudik)

## Training Configuration

| Category               | Parameter |         Value |
|------------------------|---|--------------:|
| Training               | epochs |           300 |
| Training               | batch_size |            12 |
| Training               | image_size |          1024 |
| Training               | model_name |     `yolo11m` |
| Training               | seed |            42 |
| Learning Rate          | lr0 |          0.01 |
| Learning Rate          | lrf |          0.01 |
| Augmentation           | auto_augment | `randaugment` |
| Color Augmentation     | hsv_h |          0.01 |
| Color Augmentation     | hsv_s |           0.5 |
| Color Augmentation     | hsv_v |           0.3 |
| Geometric Augmentation | translate |           0.1 |
| Geometric Augmentation | scale |           0.4 |
| Geometric Augmentation | fliplr |           0.5 |
| SAHI | use_sahi |        `True` |
| SAHI | slice_height |          2048 |
| SAHI | slice_width |           768 |
| SAHI | overlap_height_ratio |           0.2 |
| SAHI | overlap_width_ratio |           0.3 |


## Evaluation Results: 

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2200 |
| AP50 | 0.4106 |
| AP75 | 0.2110 |
| APeasy | 0.3463 |
| APmoderate | 0.2934 |
| APhard | 0.2200 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4692 |
| ARmoderate | 0.3969 |
| ARhard | 0.3030 |
| AR1 | 0.0959 |
| AR10 | 0.2634 |
| AR100 | 0.3030 |
| AR200 | 0.3030 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | road_vehicle | signal_pole | train | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6815 | 0.5299 | 0.3649 | 0.3396 | 0.2398 | 0.5723 | 0.1463 |
| 0.55 | 0.6487 | 0.4986 | 0.3381 | 0.2922 | 0.2008 | 0.5493 | 0.1125 |
| 0.60 | 0.6146 | 0.4729 | 0.2919 | 0.2260 | 0.1696 | 0.5123 | 0.0943 |
| 0.65 | 0.5654 | 0.4379 | 0.2548 | 0.1804 | 0.1491 | 0.4700 | 0.0801 |
| 0.70 | 0.4884 | 0.3904 | 0.2075 | 0.1365 | 0.1261 | 0.3924 | 0.0763 |
| 0.75 | 0.3992 | 0.3396 | 0.1773 | 0.0941 | 0.1017 | 0.3013 | 0.0637 |
| 0.80 | 0.2767 | 0.2730 | 0.1362 | 0.0519 | 0.0811 | 0.2179 | 0.0513 |
| 0.85 | 0.1403 | 0.1860 | 0.0813 | 0.0233 | 0.0586 | 0.1271 | 0.0229 |
| 0.90 | 0.0416 | 0.0923 | 0.0388 | 0.0079 | 0.0391 | 0.0524 | 0.0162 |
| 0.95 | 0.0024 | 0.0071 | 0.0098 | 0.0002 | 0.0127 | 0.0133 | 0.0096 |


#  Yolov11M trained on slcies (run_config_sahi_basic_augmentations_yolom.yaml)

## Training Configuration

| Category                   | Parameter |         Value |
|----------------------------|---|--------------:|
| Training                   | epochs |           300 |
| Training                   | batch_size |            12 |
| Training                   | image_size |          1024 |
| Training                   | model_name |     `yolo11s` |
| Training                   | seed |            42 |
| Learning Rate              | lr0 |          0.01 |
| Learning Rate              | lrf |          0.01 |
| Augmentation               | auto_augment | `randaugment` |
| Color Augmentation         | hsv_h |          0.01 |
| Color Augmentation         | hsv_s |           0.5 |
| Color Augmentation         | hsv_v |           0.3 |
| Geometric Augmentation     | translate |           0.1 |
| Geometric Augmentation     | scale |           0.4 |
| Geometric Augmentation     | fliplr |           0.5 |
| SAHI - Training /Inference | Slice_Height|          2048 |
| SAHI - Training /Inference          | SLICE_WIDTH |           768 |
| SAHI - Training  /Inference         | OVERLAP_HEIGHT_RATIO|           0.2 |
| SAHI - Training  /Inference          | OVERLAP_WIDTH_RATIO|           0.3 |

## Evaluation Results: Direct Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1794 |
| AP50 | 0.3312 |
| AP75 | 0.1701 |
| APeasy | 0.3605 |
| APmoderate | 0.2629 |
| APhard | 0.1794 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4381 |
| ARmoderate | 0.3215 |
| ARhard | 0.2205 |
| AR1 | 0.0885 |
| AR10 | 0.2028 |
| AR100 | 0.2205 |
| AR200 | 0.2205 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal_pole | signal | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5034 | 0.2158 | 0.3561 | 0.3717 | 0.3301 | 0.4123 | 0.1289 |
| 0.55 | 0.4819 | 0.1881 | 0.3437 | 0.3301 | 0.2991 | 0.3886 | 0.1195 |
| 0.60 | 0.4583 | 0.1544 | 0.3186 | 0.3078 | 0.2673 | 0.3547 | 0.1026 |
| 0.65 | 0.4265 | 0.1127 | 0.2992 | 0.2719 | 0.2302 | 0.3163 | 0.0931 |
| 0.70 | 0.3795 | 0.0802 | 0.2644 | 0.2436 | 0.1860 | 0.2631 | 0.0785 |
| 0.75 | 0.3115 | 0.0558 | 0.2350 | 0.2031 | 0.1483 | 0.1944 | 0.0427 |
| 0.80 | 0.2319 | 0.0342 | 0.1854 | 0.1428 | 0.1113 | 0.1353 | 0.0293 |
| 0.85 | 0.1392 | 0.0204 | 0.1355 | 0.0702 | 0.0760 | 0.0807 | 0.0213 |
| 0.90 | 0.0602 | 0.0112 | 0.0612 | 0.0366 | 0.0322 | 0.0303 | 0.0018 |
| 0.95 | 0.0135 | 0.0005 | 0.0122 | 0.0033 | 0.0061 | 0.0048 | 0.0018 |

## Evaluation Results: SAHI + Slice-Trained YOLOv11s

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2337 |
| AP50 | 0.4270 |
| AP75 | 0.2301 |
| APeasy | 0.3859 |
| APmoderate | 0.3190 |
| APhard | 0.2337 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5164 |
| ARmoderate | 0.4254 |
| ARhard | 0.3204 |
| AR1 | 0.1025 |
| AR10 | 0.2781 |
| AR100 | 0.3204 |
| AR200 | 0.3204 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | train | signal_pole | person | road_vehicle | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6966 | 0.5643 | 0.2050 | 0.3743 | 0.5832 | 0.4017 | 0.1640 |
| 0.55 | 0.6635 | 0.5308 | 0.1739 | 0.3176 | 0.5576 | 0.3536 | 0.1521 |
| 0.60 | 0.6377 | 0.5024 | 0.1425 | 0.2628 | 0.5163 | 0.3159 | 0.1436 |
| 0.65 | 0.5972 | 0.4743 | 0.1261 | 0.2106 | 0.4651 | 0.2616 | 0.1357 |
| 0.70 | 0.5274 | 0.4305 | 0.0956 | 0.1572 | 0.3930 | 0.2226 | 0.1211 |
| 0.75 | 0.4478 | 0.3828 | 0.0801 | 0.1158 | 0.3202 | 0.1769 | 0.0871 |
| 0.80 | 0.3310 | 0.3222 | 0.0625 | 0.0621 | 0.2253 | 0.1321 | 0.0331 |
| 0.85 | 0.1818 | 0.2353 | 0.0428 | 0.0340 | 0.1245 | 0.0843 | 0.0247 |
| 0.90 | 0.0573 | 0.1190 | 0.0248 | 0.0097 | 0.0500 | 0.0391 | 0.0133 |
| 0.95 | 0.0053 | 0.0203 | 0.0023 | 0.0017 | 0.0108 | 0.0090 | 0.0100 |



# Yolo11L with SAHI 
## Training Configuration

| Category | Parameter | Value |
|---|---|---:|
| Training | epochs | 45 |
| Training | batch_size | 12 |
| Training | image_size | 1024 |
| Training | model_name | `yolo11l` |
| Training | seed | 42 |
| Training | workers | 2 |
| Learning Rate | lr0 | 0.01 |
| Learning Rate | lrf | 0.01 |
| Augmentation | auto_augment | `randaugment` |
| Color Augmentation | hsv_h | 0.01 |
| Color Augmentation | hsv_s | 0.5 |
| Color Augmentation | hsv_v | 0.3 |
| Geometric Augmentation | translate | 0.1 |
| Geometric Augmentation | scale | 0.4 |
| Geometric Augmentation | fliplr | 0.5 |
| Blur Augmentation | blur | `False` |
| Blur Augmentation | blur_limit | `3,10` |
| Blur Augmentation | blur_p | 0.05 |
| Motion Blur Augmentation | motion_blur | `False` |
| Motion Blur Augmentation | motion_blur_limit | `3,10` |
| Motion Blur Augmentation | motion_blur_p | 0.05 |
| SAHI | use_sahi | `True` |
| SAHI | slice_height | 2048 |
| SAHI | slice_width | 768 |
| SAHI | overlap_height_ratio | 0.2 |
| SAHI | overlap_width_ratio | 0.3 |
| SAHI | min_area_ratio | 0.2 |



## Training results
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2373 |
| AP50 | 0.4309 |
| AP75 | 0.2296 |
| APeasy | 0.3874 |
| APmoderate | 0.3256 |
| APhard | 0.2373 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5229 |
| ARmoderate | 0.4367 |
| ARhard | 0.3259 |
| AR1 | 0.1033 |
| AR10 | 0.2854 |
| AR100 | 0.3259 |
| AR200 | 0.3259 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | road_vehicle | signal_pole | train | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6988 | 0.5628 | 0.4020 | 0.3769 | 0.2110 | 0.5816 | 0.1832 |
| 0.55 | 0.6642 | 0.5337 | 0.3695 | 0.3359 | 0.1633 | 0.5464 | 0.1703 |
| 0.60 | 0.6384 | 0.5069 | 0.3319 | 0.2906 | 0.1421 | 0.5094 | 0.1587 |
| 0.65 | 0.5944 | 0.4733 | 0.2864 | 0.2346 | 0.1185 | 0.4600 | 0.1374 |
| 0.70 | 0.5296 | 0.4352 | 0.2351 | 0.1744 | 0.0979 | 0.4012 | 0.1053 |
| 0.75 | 0.4568 | 0.3803 | 0.1863 | 0.1162 | 0.0742 | 0.3195 | 0.0737 |
| 0.80 | 0.3384 | 0.3170 | 0.1443 | 0.0752 | 0.0499 | 0.2263 | 0.0417 |
| 0.85 | 0.2016 | 0.2360 | 0.0822 | 0.0404 | 0.0289 | 0.1292 | 0.0364 |
| 0.90 | 0.0705 | 0.1309 | 0.0345 | 0.0160 | 0.0158 | 0.0585 | 0.0107 |
| 0.95 | 0.0051 | 0.0278 | 0.0077 | 0.0006 | 0.0016 | 0.0141 | 0.0043 |



















































# EdgeCrafter - Medium

## EdgeCrafter Configuration

### General

| Category | Parameter | Value |
|---|---|---:|
| Includes | custom dataset | `../dataset/custom.yml` |
| Includes | base config | `ecdet.yml` |
| Output | output_dir | `outputs/ecdet_m` |
| Task | task | `detection` |
| Model | model | `ECDet` |
| Criterion | criterion | `ECCriterion` |
| Postprocessing | postprocessor | `PostProcessor` |
| Training | epochs | 99 |
| Training | warmup_iter | 2000 |
| Training | lr_gamma | 0.5 |
| Evaluation | eval_spatial_size | `[640, 640]` |
| Checkpointing | checkpoint_freq | 5 |
| Loss | use_focal_loss | `True` |

### Model Architecture

| Component | Parameter | Value |
|---|---|---:|
| ECDet | backbone | `ViTAdapter` |
| ECDet | encoder | `HybridEncoder` |
| ECDet | decoder | `ECTransformer` |
| ViTAdapter | name | `ecvittplus` |
| ViTAdapter | embed_dim | 256 |
| ViTAdapter | num_heads | 4 |
| ViTAdapter | interaction_indexes | `[10, 11]` |
| ViTAdapter | weights_path | `ecvits/ecvittplus.pth` |
| HybridEncoder | in_channels | `[256, 256, 256]` |
| HybridEncoder | feat_strides | `[8, 16, 32]` |
| HybridEncoder | hidden_dim | 256 |
| HybridEncoder | depth_mult | 0.67 |
| HybridEncoder | expansion | 0.75 |
| HybridEncoder | dim_feedforward | 512 |
| HybridEncoder | act | `silu` |
| HybridEncoder | use_encoder_idx | `[2]` |
| HybridEncoder | num_encoder_layers | 1 |
| HybridEncoder | nhead | 8 |
| HybridEncoder | dropout | 0.0 |
| HybridEncoder | csp_type | `csp2` |
| HybridEncoder | fuse_op | `sum` |
| ECTransformer | feat_channels | `[256, 256, 256]` |
| ECTransformer | feat_strides | `[8, 16, 32]` |
| ECTransformer | hidden_dim | 256 |
| ECTransformer | num_levels | 3 |
| ECTransformer | num_layers | 4 |
| ECTransformer | eval_idx | -1 |
| ECTransformer | num_queries | 300 |
| ECTransformer | num_denoising | 100 |
| ECTransformer | label_noise_ratio | 0.5 |
| ECTransformer | box_noise_scale | 1.0 |
| ECTransformer | reg_max | 32 |
| ECTransformer | reg_scale | 4 |
| ECTransformer | layer_scale | 1 |
| ECTransformer | num_points | `[3, 6, 3]` |
| ECTransformer | cross_attn_method | `default` |
| ECTransformer | query_select_method | `default` |
| ECTransformer | activation | `silu` |
| ECTransformer | dim_feedforward | 1024 |
| PostProcessor | num_top_queries | 300 |

### Loss and Matching

| Component | Parameter | Value |
|---|---|---:|
| ECCriterion | weight_dict | `{loss_mal: 1, loss_bbox: 5, loss_giou: 2, loss_fgl: 0.15, loss_ddf: 1.5}` |
| ECCriterion | losses | `['mal', 'boxes', 'local']` |
| ECCriterion | gamma | 1.5 |
| ECCriterion | alpha | 0.75 |
| ECCriterion | reg_max | 32 |
| Matcher | type | `HungarianMatcher` |
| Matcher | weight_dict | `{cost_class: 2, cost_bbox: 5, cost_giou: 2}` |
| Matcher | alpha | 0.25 |
| Matcher | gamma | 2.0 |

### Optimizer and LR Schedule

| Category | Parameter | Value |
|---|---|---:|
| Optimizer | type | `AdamW` |
| Optimizer | lr | 0.0005 |
| Optimizer | betas | `[0.9, 0.999]` |
| Optimizer | weight_decay | 0.0001 |
| Optimizer param group | backbone non-norm/bn/bias lr | 0.000025 |
| Optimizer param group | backbone norm/bn/bias lr | 0.000025 |
| Optimizer param group | backbone norm/bn/bias weight_decay | 0.0 |
| Optimizer param group | non-backbone norm/bn/bias weight_decay | 0.0 |
| LR Scheduler | type | `flatcosine` |
| LR Scheduler | lr_gamma | 0.5 |
| LR Scheduler | warmup_iter | 2000 |
| LR Scheduler | flat_epoch | `~` |

### Training and Augmentation

| Category | Parameter | Value |
|---|---|---:|
| Train dataloader | total_batch_size | 24 |
| Train dataloader | num_workers | 4 |
| Train dataloader | shuffle | `True` |
| Train transforms | policy | `stop_epoch` |
| Train transforms | mosaic_epoch | 30 |
| Train transforms | mosaic_prob | 0.75 |
| Train transforms | stop_epoch | 90 |
| Train transforms | remove_ops | `['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']` |
| Collate function | type | `BatchImageCollateFunction` |
| Collate function | mixup_prob | 0.75 |
| Collate function | mixup_epoch | 20 |

### Transform Pipeline

| Order | Transform | Key Parameters |
|---:|---|---|
| 1 | `Mosaic` | `output_size=320`, `rotation_range=10`, `translation_range=[0.1, 0.1]`, `scaling_range=[0.5, 1.5]` |
| 2 | `RandomPhotometricDistort` | `p=0.5` |
| 3 | `RandomZoomOut` | `fill=0` |
| 4 | `RandomIoUCrop` | `p=0.8` |
| 5 | `SanitizeBoundingBoxes` | `min_size=1` |
| 6 | `RandomHorizontalFlip` | default |
| 7 | `Resize` | `size=[640, 640]` |
| 8 | `SanitizeBoundingBoxes` | `min_size=1` |
| 9 | `ConvertPILImage` | `dtype='float32'`, `scale=True` |
| 10 | `Normalize` | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |
| 11 | `ConvertBoxes` | `fmt='cxcywh'`, `normalize=True` |

### Validation and Runtime

| Category | Parameter | Value |
|---|---|---:|
| Validation | eval_spatial_size | `[640, 640]` |
| Val dataloader | total_batch_size | 12 |
| Val dataloader | num_workers | 4 |
| Val dataloader | shuffle | `False` |
| Val transforms | resize | `[640, 640]` |
| Runtime | print_freq | 500 |
| Runtime | sync_bn | `True` |
| Runtime | find_unused_parameters | `False` |
| Runtime | use_amp | `True` |
| Runtime | use_ema | `True` |
| EMA | decay | 0.9999 |
| EMA | warmups | 1000 |
| EMA | start | 0 |
| Runtime | clip_max_norm | 0.1 |

## Evaluation Results: EdgeCrafter Direct Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1168 |
| AP50 | 0.2407 |
| AP75 | 0.0991 |
| APeasy | 0.2295 |
| APmoderate | 0.1595 |
| APhard | 0.1168 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.3540 |
| ARmoderate | 0.2449 |
| ARhard | 0.1768 |
| AR1 | 0.0756 |
| AR10 | 0.1563 |
| AR100 | 0.1768 |
| AR200 | 0.1768 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | road_vehicle | train | signal_pole | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.2612 | 0.1577 | 0.3076 | 0.4131 | 0.0779 | 0.3474 | 0.1199 |
| 0.55 | 0.2305 | 0.1417 | 0.2671 | 0.3976 | 0.0611 | 0.2969 | 0.1121 |
| 0.60 | 0.1996 | 0.1254 | 0.2269 | 0.3565 | 0.0436 | 0.2395 | 0.0951 |
| 0.65 | 0.1707 | 0.1099 | 0.1863 | 0.3278 | 0.0306 | 0.1938 | 0.0824 |
| 0.70 | 0.1409 | 0.0922 | 0.1441 | 0.2771 | 0.0240 | 0.1448 | 0.0593 |
| 0.75 | 0.1136 | 0.0756 | 0.1074 | 0.2448 | 0.0188 | 0.1031 | 0.0302 |
| 0.80 | 0.0802 | 0.0593 | 0.0760 | 0.1939 | 0.0102 | 0.0667 | 0.0128 |
| 0.85 | 0.0481 | 0.0414 | 0.0541 | 0.1358 | 0.0028 | 0.0353 | 0.0090 |
| 0.90 | 0.0197 | 0.0193 | 0.0212 | 0.0757 | 0.0011 | 0.0154 | 0.0058 |
| 0.95 | 0.0099 | 0.0017 | 0.0109 | 0.0032 | 0.0000 | 0.0106 | 0.0034 |


## Evaluation Results: EdgeCrafter + SAHI Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2469 |
| AP50 | 0.4750 |
| AP75 | 0.2227 |
| APeasy | 0.4055 |
| APmoderate | 0.3299 |
| APhard | 0.2469 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5718 |
| ARmoderate | 0.4728 |
| ARhard | 0.3706 |
| AR1 | 0.1103 |
| AR10 | 0.3057 |
| AR100 | 0.3703 |
| AR200 | 0.3706 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.7427 | 0.4615 | 0.3734 | 0.2557 | 0.4269 | 0.6759 | 0.3891 |
| 0.55 | 0.7109 | 0.4346 | 0.3063 | 0.2189 | 0.3739 | 0.6318 | 0.3443 |
| 0.60 | 0.6723 | 0.4037 | 0.2601 | 0.1744 | 0.3182 | 0.5745 | 0.3401 |
| 0.65 | 0.6149 | 0.3643 | 0.2013 | 0.1542 | 0.2662 | 0.5117 | 0.2921 |
| 0.70 | 0.5474 | 0.3211 | 0.1417 | 0.1254 | 0.2116 | 0.4172 | 0.2229 |
| 0.75 | 0.4549 | 0.2736 | 0.0942 | 0.0973 | 0.1585 | 0.3169 | 0.1634 |
| 0.80 | 0.3354 | 0.2173 | 0.0503 | 0.0792 | 0.1138 | 0.2148 | 0.1290 |
| 0.85 | 0.1931 | 0.1552 | 0.0257 | 0.0552 | 0.0796 | 0.1369 | 0.0810 |
| 0.90 | 0.0715 | 0.0785 | 0.0140 | 0.0330 | 0.0386 | 0.0612 | 0.0316 |
| 0.95 | 0.0067 | 0.0048 | 0.0006 | 0.0065 | 0.0063 | 0.0132 | 0.0110 |




# Edgecrafter - Large
## Config (#tbd)
- Both trained with sahi


## Evaluation Results: EdgeCrafter Direct Inference

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1198 |
| AP50 | 0.2588 |
| AP75 | 0.0973 |
| APeasy | 0.2396 |
| APmoderate | 0.1671 |
| APhard | 0.1198 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.3426 |
| ARmoderate | 0.2409 |
| ARhard | 0.1736 |
| AR1 | 0.0773 |
| AR10 | 0.1574 |
| AR100 | 0.1736 |
| AR200 | 0.1736 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.2782 | 0.1817 | 0.1388 | 0.4185 | 0.3630 | 0.2899 | 0.1416 |
| 0.55 | 0.2482 | 0.1648 | 0.1102 | 0.3756 | 0.3185 | 0.2434 | 0.1213 |
| 0.60 | 0.2130 | 0.1512 | 0.0826 | 0.3293 | 0.2765 | 0.1907 | 0.1107 |
| 0.65 | 0.1808 | 0.1331 | 0.0566 | 0.2771 | 0.2310 | 0.1469 | 0.0880 |
| 0.70 | 0.1543 | 0.1133 | 0.0378 | 0.2409 | 0.1873 | 0.1063 | 0.0752 |
| 0.75 | 0.1165 | 0.0893 | 0.0211 | 0.1986 | 0.1352 | 0.0765 | 0.0440 |
| 0.80 | 0.0820 | 0.0623 | 0.0116 | 0.1673 | 0.0892 | 0.0469 | 0.0233 |
| 0.85 | 0.0490 | 0.0349 | 0.0055 | 0.0917 | 0.0536 | 0.0265 | 0.0192 |
| 0.90 | 0.0242 | 0.0102 | 0.0020 | 0.0396 | 0.0285 | 0.0127 | 0.0082 |
| 0.95 | 0.0062 | 0.0014 | 0.0000 | 0.0053 | 0.0103 | 0.0056 | 0.0082 |



## Evaluation Results: EdgeCrafterL + Oversampling +  SAHI, EVAL With SAHI + postprocessing_match_threshold=0.5 
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2726 |
| AP50 | 0.5214 |
| AP75 | 0.2457 |
| APeasy | 0.4305 |
| APmoderate | 0.3570 |
| APhard | 0.2726 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5663 |
| ARmoderate | 0.4720 |
| ARhard | 0.3742 |
| AR1 | 0.1141 |
| AR10 | 0.3165 |
| AR100 | 0.3742 |
| AR200 | 0.3742 |
### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.7994 | 0.5791 | 0.4955 | 0.2870 | 0.4631 | 0.6527 | 0.3730 |
| 0.55 | 0.7717 | 0.5447 | 0.4277 | 0.2500 | 0.4071 | 0.6175 | 0.3517 |
| 0.60 | 0.7325 | 0.5056 | 0.3619 | 0.2101 | 0.3382 | 0.5591 | 0.3146 |
| 0.65 | 0.6760 | 0.4545 | 0.2857 | 0.1721 | 0.2760 | 0.4936 | 0.2497 |
| 0.70 | 0.6032 | 0.3973 | 0.2019 | 0.1358 | 0.2178 | 0.4242 | 0.1698 |
| 0.75 | 0.5044 | 0.3439 | 0.1401 | 0.1046 | 0.1715 | 0.3307 | 0.1245 |
| 0.80 | 0.3761 | 0.2739 | 0.0870 | 0.0851 | 0.1337 | 0.2286 | 0.1111 |
| 0.85 | 0.2258 | 0.1854 | 0.0406 | 0.0559 | 0.0974 | 0.1289 | 0.0722 |
| 0.90 | 0.0869 | 0.1012 | 0.0142 | 0.0367 | 0.0541 | 0.0635 | 0.0160 |
| 0.95 | 0.0166 | 0.0145 | 0.0020 | 0.0125 | 0.0144 | 0.0192 | 0.0075 |












## Evaluation Results: EdgeCrafterL + Oversampling , EVAL With SAHI + postprocessing_match_threshold=0.5
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2637 |
| AP50 | 0.5136 |
| AP75 | 0.2383 |
| APeasy | 0.4007 |
| APmoderate | 0.3422 |
| APhard | 0.2637 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5618 |
| ARmoderate | 0.4745 |
| ARhard | 0.3809 |
| AR1 | 0.1105 |
| AR10 | 0.3184 |
| AR100 | 0.3807 |
| AR200 | 0.3809 |
### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.7844 | 0.5943 | 0.5189 | 0.2382 | 0.4468 | 0.6712 | 0.3415 |
| 0.55 | 0.7550 | 0.5634 | 0.4405 | 0.1924 | 0.3958 | 0.6257 | 0.3294 |
| 0.60 | 0.7135 | 0.5081 | 0.3693 | 0.1548 | 0.3328 | 0.5809 | 0.2752 |
| 0.65 | 0.6614 | 0.4567 | 0.2860 | 0.1222 | 0.2722 | 0.4998 | 0.2115 |
| 0.70 | 0.5864 | 0.3999 | 0.2144 | 0.0958 | 0.2117 | 0.4144 | 0.1722 |
| 0.75 | 0.4921 | 0.3462 | 0.1433 | 0.0708 | 0.1635 | 0.3241 | 0.1282 |
| 0.80 | 0.3604 | 0.2789 | 0.0754 | 0.0563 | 0.1268 | 0.2157 | 0.1065 |
| 0.85 | 0.2116 | 0.1935 | 0.0346 | 0.0309 | 0.0813 | 0.1222 | 0.0765 |
| 0.90 | 0.0768 | 0.0920 | 0.0149 | 0.0143 | 0.0418 | 0.0505 | 0.0188 |
| 0.95 | 0.0121 | 0.0194 | 0.0099 | 0.0027 | 0.0082 | 0.0111 | 0.0074 |

## Evaluation Results: EdgeCrafterL + Oversampling , EVAL With SAHI + postprocessing_match_threshold=0.3
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2603 |
| AP50 | 0.5030 |
| AP75 | 0.2390 |
| APeasy | 0.4049 |
| APmoderate | 0.3406 |
| APhard | 0.2603 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5481 |
| ARmoderate | 0.4550 |
| ARhard | 0.3598 |
| AR1 | 0.1105 |
| AR10 | 0.3124 |
| AR100 | 0.3598 |
| AR200 | 0.3598 |
### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|--------------:|---:|---:|---:|---:|---:|---:|
| 0.50 |        0.7837 | 0.5934 | 0.5127 | 0.2328 | 0.4096 | 0.6463 | 0.3421 |
| 0.55 |        0.7492 | 0.5617 | 0.4306 | 0.1773 | 0.3611 | 0.6011 | 0.3384 |
| 0.60 |        0.7060 | 0.5065 | 0.3641 | 0.1460 | 0.3082 | 0.5575 | 0.2844 |
| 0.65 |        0.6583 | 0.4580 | 0.2814 | 0.1169 | 0.2553 | 0.4834 | 0.2204 |
| 0.70 |        0.5883 | 0.4008 | 0.2142 | 0.0971 | 0.2020 | 0.4041 | 0.1755 |
| 0.75 |        0.4960 | 0.3482 | 0.1435 | 0.0712 | 0.1606 | 0.3194 | 0.1340 |
| 0.80 |        0.3631 | 0.2791 | 0.0762 | 0.0572 | 0.1268 | 0.2128 | 0.1129 |
| 0.85 |      0.2125 s | 0.1951 | 0.0350 | 0.0323 | 0.0841 | 0.1217 | 0.0808 |
| 0.90 |        0.0776 | 0.0926 | 0.0150 | 0.0147 | 0.0447 | 0.0515 | 0.0205 |
| 0.95 |        0.0120 | 0.0195 | 0.0099 | 0.0027 | 0.0088 | 0.0110 | 0.0081 |


## Evaluation Results: EdgeCrafterL + Oversampling , EVAL With SAHI + postprocessing_match_threshold=0.7
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2672 |
| AP50 | 0.5066 |
| AP75 | 0.2467 |
| APeasy | 0.4004 |
| APmoderate | 0.3440 |
| APhard | 0.2672 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5841 |
| ARmoderate | 0.4984 |
| ARhard | 0.4065 |
| AR1 | 0.1105 |
| AR10 | 0.3276 |
| AR100 | 0.4056 |
| AR200 | 0.4065 |
### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.7636 | 0.5750 | 0.5142 | 0.2382 | 0.4475 | 0.6617 | 0.3459 |
| 0.55 | 0.7408 | 0.5459 | 0.4415 | 0.1953 | 0.4029 | 0.6281 | 0.3402 |
| 0.60 | 0.7050 | 0.5039 | 0.3782 | 0.1605 | 0.3474 | 0.5901 | 0.2890 |
| 0.65 | 0.6648 | 0.4587 | 0.2922 | 0.1298 | 0.2911 | 0.5259 | 0.2296 |
| 0.70 | 0.5929 | 0.3997 | 0.2216 | 0.1016 | 0.2296 | 0.4427 | 0.1753 |
| 0.75 | 0.4990 | 0.3449 | 0.1479 | 0.0743 | 0.1798 | 0.3471 | 0.1336 |
| 0.80 | 0.3652 | 0.2757 | 0.0774 | 0.0591 | 0.1366 | 0.2289 | 0.1127 |
| 0.85 | 0.2159 | 0.1922 | 0.0348 | 0.0312 | 0.0854 | 0.1299 | 0.0794 |
| 0.90 | 0.0788 | 0.0921 | 0.0148 | 0.0142 | 0.0423 | 0.0545 | 0.0174 |
| 0.95 | 0.0123 | 0.0200 | 0.0099 | 0.0027 | 0.0089 | 0.0112 | 0.0065 |


## Evaluation Results: EdgeCrafterL + Oversampling , EVAL With SAHI + postprocessing_match_threshold=0.9
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2423 |
| AP50 | 0.4385 |
| AP75 | 0.2340 |
| APeasy | 0.3723 |
| APmoderate | 0.3159 |
| APhard | 0.2423 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.6016 |
| ARmoderate | 0.5187 |
| ARhard | 0.4291 |
| AR1 | 0.1105 |
| AR10 | 0.3214 |
| AR100 | 0.4264 |
| AR200 | 0.4291 |
### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | train | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6631 | 0.4874 | 0.4147 | 0.2129 | 0.3941 | 0.5824 | 0.3150 |
| 0.55 | 0.6470 | 0.4697 | 0.3672 | 0.1788 | 0.3560 | 0.5548 | 0.2939 |
| 0.60 | 0.6251 | 0.4374 | 0.3213 | 0.1506 | 0.3121 | 0.5239 | 0.2560 |
| 0.65 | 0.5936 | 0.4067 | 0.2599 | 0.1210 | 0.2716 | 0.4749 | 0.2056 |
| 0.70 | 0.5419 | 0.3653 | 0.2043 | 0.0949 | 0.2168 | 0.4091 | 0.1606 |
| 0.75 | 0.4664 | 0.3244 | 0.1419 | 0.0740 | 0.1741 | 0.3324 | 0.1249 |
| 0.80 | 0.3560 | 0.2724 | 0.0773 | 0.0602 | 0.1346 | 0.2324 | 0.1122 |
| 0.85 | 0.2170 | 0.1966 | 0.0373 | 0.0307 | 0.0902 | 0.1375 | 0.0810 |
| 0.90 | 0.0832 | 0.0984 | 0.0153 | 0.0134 | 0.0446 | 0.0572 | 0.0146 |
| 0.95 | 0.0127 | 0.0199 | 0.0099 | 0.0024 | 0.0096 | 0.0116 | 0.0051 |


## Evaluation Results: EdgeCrafterL + Oversampling , EVAL With SAHI + postprocessing_match_threshold=0.7, GREEDYNNM
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2487 |
| AP50 | 0.4981 |
| AP75 | 0.2135 |
| APeasy | 0.3663 |
| APmoderate | 0.3182 |
| APhard | 0.2487 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5491 |
| ARmoderate | 0.4728 |
| ARhard | 0.3857 |
| AR1 | 0.1036 |
| AR10 | 0.3107 |
| AR100 | 0.3847 |
| AR200 | 0.3857 |
### AP by Class per IoU Threshold

| IoU | train | catenary_pole | signal_pole | signal | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.2251 | 0.7572 | 0.5087 | 0.5746 | 0.4314 | 0.6523 | 0.3371 |
| 0.55 | 0.1887 | 0.7338 | 0.4414 | 0.5451 | 0.3786 | 0.6158 | 0.3281 |
| 0.60 | 0.1446 | 0.6952 | 0.3655 | 0.5012 | 0.3240 | 0.5715 | 0.2689 |
| 0.65 | 0.1121 | 0.6436 | 0.2688 | 0.4445 | 0.2637 | 0.4963 | 0.1994 |
| 0.70 | 0.0895 | 0.5728 | 0.2079 | 0.3645 | 0.2004 | 0.4066 | 0.1504 |
| 0.75 | 0.0603 | 0.4748 | 0.1237 | 0.2845 | 0.1373 | 0.2956 | 0.1183 |
| 0.80 | 0.0431 | 0.3500 | 0.0685 | 0.1922 | 0.0891 | 0.1937 | 0.0877 |
| 0.85 | 0.0214 | 0.2170 | 0.0275 | 0.0963 | 0.0490 | 0.0989 | 0.0721 |
| 0.90 | 0.0109 | 0.0915 | 0.0139 | 0.0409 | 0.0179 | 0.0414 | 0.0293 |
| 0.95 | 0.0021 | 0.0144 | 0.0099 | 0.0048 | 0.0034 | 0.0075 | 0.0111 |