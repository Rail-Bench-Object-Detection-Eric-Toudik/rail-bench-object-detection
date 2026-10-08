
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


# Edgecrafter L - NO SAHI (inference and training)
eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/79l7jzs8_model:v0
## Evaluation Results: EdgeCrafterL + Oversampling  (NO SAHI!)

| Metric | Value |
|---|---:|
| AP | 0.1983 |
| AP50 | 0.3916 |
| AP75 | 0.1706 |
| APeasy | 0.3662 |
| APmoderate | 0.2718 |
| APhard | 0.1983 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.4636 |
| ARmoderate | 0.3506 |
| ARhard | 0.2594 |
| AR1 | 0.1034 |
| AR10 | 0.2205 |
| AR100 | 0.2588 |
| AR200 | 0.2594 |
### AP by Class per IoU Threshold

| IoU | catenary_pole | person | signal_pole | signal | train | road_vehicle | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.4432 | 0.5051 | 0.2007 | 0.3483 | 0.5299 | 0.4636 | 0.2504 |
| 0.55 | 0.4125 | 0.4503 | 0.1657 | 0.3251 | 0.5037 | 0.4210 | 0.2246 |
| 0.60 | 0.3729 | 0.3872 | 0.1304 | 0.2928 | 0.4695 | 0.3775 | 0.1782 |
| 0.65 | 0.3285 | 0.3202 | 0.0960 | 0.2527 | 0.4188 | 0.3199 | 0.1440 |
| 0.70 | 0.2755 | 0.2347 | 0.0629 | 0.2194 | 0.3647 | 0.2473 | 0.1071 |
| 0.75 | 0.2184 | 0.1666 | 0.0382 | 0.1873 | 0.3321 | 0.1883 | 0.0630 |
| 0.80 | 0.1553 | 0.0916 | 0.0225 | 0.1414 | 0.2771 | 0.1318 | 0.0322 |
| 0.85 | 0.0942 | 0.0520 | 0.0100 | 0.0990 | 0.2359 | 0.0866 | 0.0230 |
| 0.90 | 0.0375 | 0.0222 | 0.0031 | 0.0404 | 0.1381 | 0.0419 | 0.0052 |
| 0.95 | 0.0112 | 0.0060 | 0.0007 | 0.0048 | 0.0621 | 0.0121 | 0.0035 |
# Edgecrafter - Large SAHI
## Config (#tbd)
- Both trained with sahi
eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/upva347y_model:v0

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



# Hybrid Approach - Large EdgeCrafter Model with and without SAHI
- Direct:  eric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/79l7jzs8_model:v0
- SAHI: ric-walsh-fau/Rail-Bench-Object-Detection-EdgeCrafter/upva347y_model:v0
- Mergin strategy:
 - Both models predict all, but we keep only "trains" from direct and everything except trains from the SAHI model
### Overall AP Metrics

| Metric | Value |
|---|---:|
| AP | 0.2968 |
| AP50 | 0.5429 |
| AP75 | 0.2773 |
| APeasy | 0.4765 |
| APmoderate | 0.3885 |
| APhard | 0.2968 |
### Recall Metrics

| Metric | Value |
|---|---:|
| AReasy | 0.5782 |
| ARmoderate | 0.4720 |
| ARhard | 0.3691 |
| AR1 | 0.1292 |
| AR10 | 0.3233 |
| AR100 | 0.3691 |
| AR200 | 0.3691 |
### AP by Class per IoU Threshold

| IoU | train | catenary_pole | signal | signal_pole | road_vehicle | person | bicycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.5299 | 0.8010 | 0.5729 | 0.4922 | 0.4071 | 0.6293 | 0.3682 |
| 0.55 | 0.5037 | 0.7716 | 0.5390 | 0.4278 | 0.3597 | 0.5959 | 0.3564 |
| 0.60 | 0.4695 | 0.7323 | 0.5002 | 0.3615 | 0.3053 | 0.5443 | 0.3061 |
| 0.65 | 0.4188 | 0.6796 | 0.4522 | 0.2847 | 0.2554 | 0.4844 | 0.2380 |
| 0.70 | 0.3647 | 0.6104 | 0.3956 | 0.2028 | 0.2054 | 0.4152 | 0.1641 |
| 0.75 | 0.3321 | 0.5101 | 0.3442 | 0.1405 | 0.1638 | 0.3271 | 0.1232 |
| 0.80 | 0.2771 | 0.3796 | 0.2757 | 0.0872 | 0.1307 | 0.2253 | 0.1092 |
| 0.85 | 0.2359 | 0.2277 | 0.1865 | 0.0407 | 0.0956 | 0.1304 | 0.0731 |
| 0.90 | 0.1381 | 0.0879 | 0.1018 | 0.0142 | 0.0524 | 0.0635 | 0.0164 |
| 0.95 | 0.0621 | 0.0167 | 0.0145 | 0.0020 | 0.0148 | 0.0196 | 0.0078 |