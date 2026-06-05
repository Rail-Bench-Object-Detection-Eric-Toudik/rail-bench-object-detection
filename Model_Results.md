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

| Category | Parameter | Value |
|---|---|---:|
| Training | epochs | 60 |
| Training | batch_size | 12 |
| Training | image_size | 1024 |
| Training | model_name | `yolo11s` |
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
| SAHI | overlap_width_ratio | 0.2 |



## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1278 |
| AP50 | 0.2402 |
| AP75 | 0.1189 |
| APeasy | 0.2740 |
| APmoderate | 0.1889 |
| APhard | 0.1278 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.3388 |
| ARmoderate | 0.2335 |
| ARhard | 0.1561 |
| AR1 | 0.0670 |
| AR10 | 0.1471 |
| AR100 | 0.1561 |
| AR200 | 0.1561 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | signal_pole | person | train | road_vehicle | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.3985 | 0.2391 | 0.1510 | 0.2891 | 0.3130 | 0.2433 | 0.0473 |
| 0.55 | 0.3786 | 0.2276 | 0.1294 | 0.2754 | 0.2902 | 0.2247 | 0.0473 |
| 0.60 | 0.3568 | 0.2119 | 0.1016 | 0.2494 | 0.2514 | 0.1990 | 0.0437 |
| 0.65 | 0.3238 | 0.1959 | 0.0763 | 0.2153 | 0.2322 | 0.1628 | 0.0332 |
| 0.70 | 0.2813 | 0.1812 | 0.0566 | 0.1762 | 0.1935 | 0.1322 | 0.0332 |
| 0.75 | 0.2273 | 0.1559 | 0.0372 | 0.1325 | 0.1565 | 0.1039 | 0.0192 |
| 0.80 | 0.1629 | 0.1308 | 0.0233 | 0.0832 | 0.1062 | 0.0817 | 0.0119 |
| 0.85 | 0.0956 | 0.0894 | 0.0127 | 0.0455 | 0.0497 | 0.0523 | 0.0079 |
| 0.90 | 0.0367 | 0.0421 | 0.0104 | 0.0185 | 0.0287 | 0.0250 | 0.0079 |
| 0.95 | 0.0109 | 0.0046 | 0.0000 | 0.0012 | 0.0056 | 0.0047 | 0.0020 |


# Trained on image slices, inference sliced (baseline_yolo11s_60epochs_SAHI_Training_1024_eric)
## Training Configuration

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
| SAHI | overlap_width_ratio  (train)            |           0.2 |
| SAHI | overlap_width_ratio  (inference)        |           0.3 |


## Evaluation Results

### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.1922 |
| AP50 | 0.3611 |
| AP75 | 0.1798 |
| APeasy | 0.3549 |
| APmoderate | 0.2779 |
| APhard | 0.1922 |

### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4638 |
| ARmoderate | 0.3642 |
| ARhard | 0.2565 |
| AR1 | 0.0846 |
| AR10 | 0.2289 |
| AR100 | 0.2565 |
| AR200 | 0.2565 |

### AP by Class per IoU Threshold

| IoU | catenary_pole | signal | train | road_vehicle | signal_pole | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6009 | 0.4459 | 0.2090 | 0.3408 | 0.3012 | 0.4989 | 0.1309 |
| 0.55 | 0.5764 | 0.4265 | 0.1794 | 0.3040 | 0.2574 | 0.4738 | 0.1230 |
| 0.60 | 0.5444 | 0.4021 | 0.1501 | 0.2701 | 0.2097 | 0.4364 | 0.1158 |
| 0.65 | 0.5030 | 0.3820 | 0.1219 | 0.2278 | 0.1514 | 0.3779 | 0.1043 |
| 0.70 | 0.4399 | 0.3442 | 0.0875 | 0.1898 | 0.0999 | 0.3174 | 0.0823 |
| 0.75 | 0.3576 | 0.3082 | 0.0771 | 0.1512 | 0.0652 | 0.2483 | 0.0508 |
| 0.80 | 0.2595 | 0.2616 | 0.0582 | 0.1177 | 0.0409 | 0.1546 | 0.0389 |
| 0.85 | 0.1341 | 0.1955 | 0.0416 | 0.0739 | 0.0157 | 0.0905 | 0.0273 |
| 0.90 | 0.0403 | 0.1017 | 0.0082 | 0.0337 | 0.0066 | 0.0310 | 0.0066 |
| 0.95 | 0.0031 | 0.0126 | 0.0010 | 0.0050 | 0.0003 | 0.0066 | 0.0015 |


