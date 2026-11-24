# RAIL-BENCH Object Detection Challenge

**Small-object detection in railway scenes with YOLO11, EdgeCrafter (ECDet), sliced training/inference (SAHI) and class-specific hybrid inference**

Project report at the Pattern Recognition Lab (Lehrstuhl für Mustererkennung, Informatik 5), Department Informatik, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

| |                                                                                                                            |
|---|----------------------------------------------------------------------------------------------------------------------------|
| **Authors** | Eric Walsh, Toudik Isavi Arktan                                                                                            |
| **Advisor** | Dr.-Ing. Vincent Christlein                                                                                                |
| **Project period** | 15.04.2026 – 16.11.2026                                                                                                    |
| **Challenge** | [RAIL-BENCH Object Detection](https://www.mrt.kit.edu/railbench/object_detection) (Karlsruhe Institute of Technology, KIT) |

---

## Summary

This repository accompanies our participation in the RAIL-BENCH Object Detection Challenge. The goal was to develop, train and evaluate an object detector for railway scenes recorded from a train driver's perspective, with a focus on small and distant objects.

We compared **YOLO11** (S/M/L) with the transformer-based **EdgeCrafter ECDet** (M/L) and investigated data augmentation, class oversampling, image slicing during training and inference (**SAHI**), input resolution and the IoU threshold for Non-Maximum Suppression (NMS). Our best configuration is a **class-specific hybrid inference** that combines two ECDet-L models:

- a model trained on **full images** with direct inference (used for the class `train`), and
- a model trained on **image slices** with SAHI inference (used for the remaining six classes).

### Key results

| Setting | AP hard |
|---|---|
| YOLO11s baseline (no augmentation, validation) | 14.60 % |
| Best YOLO11 (YOLO11l, sliced training + SAHI, validation) | 23.73 % |
| ECDet-L, sliced training + SAHI (1024 px, validation) | 31.86 % |
| **ECDet-L hybrid inference (1024 px, validation)** | **36.02 %** |
| **ECDet-L hybrid inference (test set, official submission)** | **34.19 %** |

Our submission ranked **first of three submissions** on the RAIL-BENCH Object Detection leaderboard as of **12 November 2026**. Since the challenge is open-ended, this ranking is a snapshot in time.

---

## Table of Contents

1. [Challenge and Dataset](#challenge-and-dataset)
2. [Methods](#methods)
3. [Experimental Protocol](#experimental-protocol)
4. [Results](#results)
5. [Discussion and Limitations](#discussion-and-limitations)
6. [Using the Hybrid Model (Inference)](#using-the-hybrid-model-inference)
7. [Acknowledgements](#acknowledgements)
8. [References](#references)
9. [Citation](#citation)

---

## Challenge and Dataset

RAIL-BENCH is a benchmark suite for perception tasks in the railway domain, introduced by KIT. It comprises five challenges: Rail Track Detection, Object Detection, Vegetation Segmentation, Multiple Object Tracking and Monocular Visual Odometry [Bät+26]. Motivation and background: conventional railway networks contain level crossings, open platform areas and shared-traffic environments, so a free track cannot be assumed at all times, which makes reliable perception a prerequisite for future Automatic Train Operation (ATO).

**Object Detection Challenge.** The dataset consists of 2,500 real-world RGB images recorded from a train driver's perspective under diverse conditions (urban, rural and station scenes; varying weather and camera setups). According to the challenge website, it is split into 1,500 training, 500 validation and 500 test images. Images are annotated with axis-aligned bounding boxes for seven classes:

`train` · `person` · `bicycle` · `road_vehicle` · `signal` · `signal_pole` · `catenary_pole`

Annotations additionally carry an occlusion level (0 = no occlusion to 4 = heavy occlusion) and ignore flags.

**Metric.** Evaluation uses an adapted, COCO-style Average Precision computed over multiple IoU thresholds [Bät+26]. Each annotation is assigned to one of three difficulty levels (easy, moderate, hard) based on bounding-box size and occlusion. The ranking metric is **AP hard**, which considers all ground-truth boxes including those of the easy and moderate levels. All metrics are computed with the official [RAIL-BENCH toolkit](https://github.com/KIT-MRT/RAIL-BENCH_toolkit).

### Exploratory data analysis

The annotations contain no empty boxes and no boxes outside the image frame. Images are high-resolution; 3840×2160 and 4112×2504 are the most common sizes.

| Class | Share of annotations (train) | Share of annotations (val) | Median box area relative to image |
|---|---|---|---|
| catenary_pole | 44.1 % | 44.8 % | 0.0426 % |
| person | 15.7 % | 18.3 % | 0.0616 % |
| signal | 14.7 % | 12.0 % | 0.00525 % |
| road_vehicle | 10.8 % | 10.4 % | 0.108 % |
| signal_pole | 10.1 % | 10.0 % | 0.0113 % |
| train | 3.2 % | 3.0 % | 0.479 % |
| bicycle | 1.3 % | 1.5 % | 0.0855 % |

Two properties shaped our methodology: (1) a strong **class imbalance** (`train` and `bicycle` are rare) and (2) very **small objects** relative to the image size (median box areas far below 1 % of the image for most classes). Road vehicles are occluded more frequently while most infrastructure objects are fully visible.

---

## Methods

### YOLO11

YOLO11 [Ult26; Kha+24] was chosen as a robust and reproducible baseline from the YOLO family. Pretrained Ultralytics weights were fine-tuned for the **S**, **M** and **L** variants. Experiments covered no augmentation, basic augmentation (HSV transformations, scaling, horizontal flipping) and additional blur / motion-blur augmentation. Oversampling was tested by duplicating images that contain underrepresented classes.

### EdgeCrafter (ECDet)

EdgeCrafter [Liu+26] is a framework of compact Vision Transformers for edge dense prediction. Its ECDet detector uses a distilled ECViT backbone (obtained by task-specialized knowledge distillation from a DINOv3-based teacher), a multi-scale feature representation built from the final transformer layers, and an RT-DETR-style hybrid encoder with a DETR-style decoder. We fine-tuned the pretrained **ECDet-M** and **ECDet-L** models using the official training scripts, and applied the same optimization strategies as for YOLO11 where possible (data augmentation, SAHI, oversampling). The input resolution was increased from 640×640 to 1024×1024 pixels.

### Sliced training and SAHI inference

Slicing Aided Hyper Inference (SAHI) [Aky+22; Aky+21] splits a large image into overlapping slices, runs detection on each slice and merges the predictions. This increases the relative size of small objects. We used the official SAHI Python library, both for inference only and in combination with **training on sliced images**. Because catenary and signal poles are vertically elongated, we additionally tested tall slices (768×2048 px).

### Class-specific hybrid inference

The full-image model with direct inference and the slice-trained model with SAHI inference showed complementary strengths: slicing helped for small infrastructure objects, while full-image inference was better for large objects. The hybrid strategy keeps the **`train` predictions from the direct (full-image) model** and takes the predictions for the **remaining six classes from the SAHI (slice-trained) model**.

---

## Experimental Protocol

- All experiments were evaluated on the RAIL-BENCH **validation set** with the RAIL-BENCH toolkit; the original training/validation split was kept throughout.
- The primary comparison metric is **AP hard**.
- Compared factors: model capacity (YOLO11 S/M/L, ECDet M/L), augmentation, SAHI slice dimensions, sliced vs. full-image training, oversampling, input resolution (640 vs. 1024 px), NMS IoU threshold (0.3, 0.5, 0.7, 0.9) and hybrid inference.
- Maximum training epochs: 300 for YOLO11 and 99 for ECDet (see the result tables).
- The final hybrid strategy was applied to the test set and submitted to the challenge.

---

## Results

All AP values are on the RAIL-BENCH validation set unless stated otherwise.

### YOLO11

| Model | Training | Inference | AP hard | AP50 | AP75 |
|---|---|---|---|---|---|
| YOLO11s | no augmentation | direct | 14.60 | 28.52 | 12.93 |
| YOLO11s | basic augmentation + blur | direct | 16.96 | 32.24 | 15.74 |
| YOLO11s | basic augmentation | direct | 17.28 | 32.83 | 15.49 |
| YOLO11s | basic augmentation, oversampling | direct | 16.08 | 30.51 | 14.73 |
| YOLO11s | basic augmentation | SAHI (768×1536) | 15.28 | 31.29 | 12.83 |
| YOLO11s | basic augmentation | SAHI (1024×1024) | 18.70 | 37.74 | 16.61 |
| YOLO11s | basic augmentation | SAHI (768×2048) | 19.41 | 37.75 | 17.52 |
| YOLO11s | basic augmentation, sliced (768×2048) | direct | 17.94 | 33.12 | 17.01 |
| YOLO11s | basic augmentation, sliced (768×2048) | SAHI (768×2048) | 21.28 | 39.85 | 19.97 |
| YOLO11m | basic augmentation | direct | 20.40 | 37.30 | 19.25 |
| YOLO11m | basic augmentation | SAHI (768×2048) | 22.00 | 41.06 | 21.10 |
| YOLO11m | basic augmentation, sliced (768×2048) | SAHI (768×2048) | 23.37 | 42.70 | 23.01 |
| YOLO11l | basic augmentation, sliced (768×2048) | SAHI (768×2048) | 23.73 | 43.09 | 22.96 |

All YOLO11 runs: resolution 1024, max. 300 epochs.

### ECDet at 640 px

| Model | Training | Inference | NMS IoU | AP hard | AP50 | AP75 |
|---|---|---|---|---|---|---|
| ECDet-M | std. aug., oversampling, sliced | direct | – | 11.68 | 24.07 | 9.91 |
| ECDet-M | std. aug., oversampling, sliced | SAHI | 0.5 | 24.69 | 47.50 | 22.27 |
| ECDet-L | std. aug., oversampling | direct | – | 19.83 | 39.16 | 17.06 |
| ECDet-L | std. aug., sliced | direct | – | 11.98 | 25.88 | 9.73 |
| ECDet-L | std. aug., sliced | SAHI | 0.5 | 27.26 | 52.14 | 24.57 |
| ECDet-L | std. aug., oversampling, sliced | SAHI | 0.3 | 26.03 | 50.30 | 23.90 |
| ECDet-L | std. aug., oversampling, sliced | SAHI | 0.5 | 26.37 | 51.36 | 23.83 |
| ECDet-L | std. aug., oversampling, sliced | SAHI | 0.7 | 26.72 | 50.66 | 24.67 |
| ECDet-L | std. aug., oversampling, sliced | SAHI | 0.9 | 24.23 | 43.85 | 23.40 |

"Sliced" = trained on 768×2048 slices; SAHI slice size 768×2048. Max. 99 epochs.

### ECDet-L at 1024 px

| Training | Inference | NMS IoU | AP hard | AP50 | AP75 |
|---|---|---|---|---|---|
| std. aug. | direct | – | 27.64 | 51.08 | 25.79 |
| std. aug., oversampling | direct | – | 26.90 | 49.95 | 25.18 |
| std. aug., sliced | direct | – | 17.18 | 34.76 | 15.04 |
| std. aug., sliced | SAHI | 0.5 | 31.86 | 57.88 | 30.44 |
| std. aug., oversampling, sliced | direct | – | 16.38 | 33.33 | 13.90 |
| std. aug., oversampling, sliced | SAHI | 0.5 | 31.23 | 57.61 | 29.54 |

Max. 99 epochs.

### Class-specific behaviour (ECDet-L, 1024 px, AP50)

| Class | Full / direct (%) | Sliced / SAHI (%) | Δ (pp) |
|---|---|---|---|
| Catenary pole | 59.16 | 84.47 | +25.31 |
| Signal pole | 28.95 | 63.59 | +34.64 |
| Signal | 46.63 | 67.96 | +21.33 |
| Person | 64.32 | 70.86 | +6.54 |
| Bicycle | 37.44 | 43.57 | +6.13 |
| Road vehicle | 59.02 | 49.82 | −9.20 |
| Train | 62.05 | 24.88 | −37.17 |

### Direct vs. SAHI vs. hybrid inference (ECDet-L, 1024 px)

| Inference | AP hard | AP50 | AP75 | AP easy | AP moderate |
|---|---|---|---|---|---|
| Direct | 27.64 | 51.08 | 25.79 | 47.54 | 36.87 |
| SAHI | 31.86 | 57.88 | 30.44 | 45.97 | 39.69 |
| **Hybrid** | **36.02** | **63.19** | **35.12** | **52.81** | **44.70** |

The hybrid configuration at 640 px reached 29.68 % AP hard (AP50 54.29, AP75 27.73). The post-processing match threshold was 0.5 for both resolutions.

### RAIL-BENCH test-set leaderboard (as of 12 November 2026)

| Submission | mAP hard | mAP moderate | mAP easy |
|---|---|---|---|
| **Ours** | **34.19** | **44.75** | **52.74** |
| junfeng2 | 22.19 | 29.43 | 35.05 |
| annikabaetz | 16.72 | 25.55 | 35.12 |

Our test result (34.19 %) is slightly below the best validation result (36.02 %), which suggests that the approach retains much of its performance on unseen images.

---

## Discussion and Limitations

**Main findings**

- Basic augmentation improved the YOLO11s baseline (14.60 → 17.28 % AP hard); additional blur augmentations gave no further gain.
- Sliced training combined with SAHI inference substantially improved results for both YOLO11 and ECDet, especially for small infrastructure objects (catenary poles, signal poles, signals). Taller slices (768×2048) outperformed square slices (1024×1024) for YOLO11s, whereas shorter tall slices (768×1536) performed worse than square slicing, so slice orientation alone does not explain the gain.
- Larger model capacity helped (YOLO11s → m → l), and ECDet-L at 1024 px with sliced training and SAHI (31.86 %) outperformed the best YOLO11 model (23.73 %) by 8.13 percentage points.
- SAHI was **not beneficial for all classes**: `train` (and, to a lesser degree, `road_vehicle`) performed better with full-image training and direct inference. The class-specific hybrid inference exploits this and gives the best overall result.
- Our simple oversampling strategy (duplicating whole images that contain underrepresented classes) did not improve overall performance for ECDet-L; the effect per class was mixed. A possible explanation is that whole images are duplicated together with all their labels instead of oversampling individual objects.
- An NMS IoU threshold of 0.7 gave the best AP hard among the tested thresholds for ECDet-L (sliced, oversampling, 640 px); AR100 rose from 35.98 % (0.3) to 42.64 % (0.9), illustrating the trade-off between suppressing duplicates and retaining valid detections.


---

## Using the Hybrid Model (Inference)

The script [`hybrid_inference.py`](hybrid_inference.py) runs the hybrid model of this project on a directory of images and writes a `pred_test.json` in the format of the official [RAIL-BENCH toolkit](https://github.com/KIT-MRT/RAIL-BENCH_toolkit) for Object Detection. It contains only the hybrid inference (no training, no YOLO11, no experiment tracking).

### The two checkpoints

| Checkpoint | Model | Training | Inference | Used for |
|---|---|---|---|---|
| `ecdet_l_1024_direct.pth` | ECDet-L, 1024 × 1024 | full images, no oversampling | direct (whole image) | class `train` |
| `ecdet_l_1024_sahi.pth` | ECDet-L, 1024 × 1024 | 768 × 2048 image slices, no oversampling | SAHI (sliced) | all other six classes |

<!--
TODO: Optional provenance (internal Weights & Biases artifacts):
direct = tyrbjv6w_model:v0, sahi = j3qccb4z_model:v0
-->

### 1. Requirements

- Python environment with the dependencies of this repository (the HPC scripts use `uv sync --frozen`).
- A GPU is recommended. The script falls back to the CPU if CUDA is not available, but sliced inference on high-resolution images is slow on a CPU.
- The script imports the ECDet model code from `src/ecdet.py` and reads the ECDet configuration `config/ecdet_large_1024_config.yaml`, so it has to be run from a checkout of this repository.

### 2. Download the weights

Download the two checkpoints and place them in the folder `weights/`:

```
weights/
├── ecdet_l_1024_direct.pth
└── ecdet_l_1024_sahi.pth
```

<!-- TODO: add the download links (e.g. GitHub Release, Hugging Face, Zenodo) once the weights are published. -->

*Download links: to be added.*

Other locations can be passed with `--direct-checkpoint` and `--sahi-checkpoint`.

### 3. Prepare the images

The input is a directory with `.jpg`, `.jpeg` or `.png` images. The images of the RAIL-BENCH Object dataset can be downloaded from the [challenge website](https://www.mrt.kit.edu/railbench/object_detection). The dataset is split into 1,500 training, 500 validation and 500 test images; the ground truth of the test set is not public.

### 4. Run the hybrid model

```bash
python hybrid_inference.py --image-dir path/to/images
```

This writes `pred_test.json` to the current directory. All options:

| Option | Default | Description |
|---|---|---|
| `--image-dir` | *(required)* | Directory with the input images |
| `--direct-checkpoint` | `weights/ecdet_l_1024_direct.pth` | Checkpoint trained on full images (class `train`) |
| `--sahi-checkpoint` | `weights/ecdet_l_1024_sahi.pth` | Checkpoint trained on slices (all other classes) |
| `--config` | `config/ecdet_large_1024_config.yaml` | ECDet configuration |
| `--output` | `pred_test.json` | Output file |
| `--device` | `cuda:0` | Torch device (e.g. `cpu`) |
| `--label-indexing` | `auto` | Class index of the model outputs: `one_based` (1..7) or `zero_based` (0..6), `auto` automatically selects the right indexing. The output file always uses 1..7. |
| `--nms-threshold` | `0.5` | IOU threshold of the NMS that merges the predictions of overlapping slices |
| `--score-threshold` | `0` | Predictions with a lower score are not written to the output file (`0` = no filtering) |

Fixed parameters (see the constants at the top of the script): slices of 768 × 2048 pixels (width × height) with an overlap of 0.3 (width) and 0.2 (height), model confidence threshold 0.001.

> **Note:** `src/run_inference.py` uses an NMS threshold of 0.3 and discards predictions with a score below 0.2 when writing its result file. `hybrid_inference.py` defaults to 0.5 and 0 (no filtering); use `--nms-threshold 0.3 --score-threshold 0.2` to apply the values of `src/run_inference.py`.

### 5. Output format

`pred_test.json` is a list with one entry per detection:

```json
[
    {
        "file_name": "image_001.jpg",
        "category_id": 1,
        "bbox": [x, y, width, height],
        "score": 0.93
    }
]
```

`bbox` is given in pixels (top-left corner, width, height). The class ids are one-based:

| `category_id` | Class |
|---|---|
| 1 | train |
| 2 | catenary_pole |
| 3 | signal_pole |
| 4 | signal |
| 5 | person |
| 6 | road_vehicle |
| 7 | bicycle |

### 6. Evaluate on the validation set

For the validation split, run the script on the validation images and use the toolkit's evaluation (data layout as described in the [toolkit README](https://github.com/KIT-MRT/RAIL-BENCH_toolkit)):

```
external/RAIL-BENCH_toolkit/Benchmarks/RAILBENCH_Object/data/
└── my_project/
    └── detectors/
        └── hybrid_ecdet_l/
            └── pred_val.json        # output of hybrid_inference.py
```

```bash
python hybrid_inference.py --image-dir ./data/images/val --split val
```

The results are written to the folder `results` in the project folder `my_project`. If the evaluation reports category mismatches, check that the category ids of the ground truth are 1–7: our own evaluation pipeline converted the ground-truth ids from 0–6 to 1–7 when necessary.

<!--
TODO (to be completed by the authors – not covered by the project report):
- Repository structure (directories, scripts, configs)
- Environment / installation instructions and dependency versions
- Instructions for data preparation (RAIL-BENCH dataset download and format conversion)
- Commands for training (YOLO11, ECDet) and sliced training
- License
-->

### External resources

The project builds on the following external resources:

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- EdgeCrafter / ECDet, official training scripts and pretrained weights [Liu+26]
- [SAHI](https://doi.org/10.5281/zenodo.5718950) Python library [Aky+21]
- Official RAIL-BENCH toolkit and dataset ([RAIL-BENCH Object Detection](https://www.mrt.kit.edu/railbench/object_detection))

---

## Acknowledgements

This project was carried out at the Pattern Recognition Lab (Lehrstuhl für Mustererkennung, Informatik 5), FAU Erlangen-Nürnberg, under the supervision of Dr.-Ing. Vincent Christlein. The RAIL-BENCH benchmark, datasets and evaluation toolkit are provided by the Institute of Measurement and Control Systems, Karlsruhe Institute of Technology (KIT).

---

## References

- **[Aky+21]** F. C. Akyon, C. Cengiz, S. O. Altinuc, D. Cavusoglu, K. Sahin, O. Eryuksel, O. Sezer. *SAHI: A lightweight vision library for performing large scale object detection and instance segmentation*, 2021. DOI: [10.5281/zenodo.5718950](https://doi.org/10.5281/zenodo.5718950)
- **[Aky+22]** F. C. Akyon, S. Onur Altinuc, A. Temizel. *Slicing aided hyper inference and fine-tuning for small object detection*. In: 2022 IEEE International Conference on Image Processing (ICIP), pp. 966–970. DOI: [10.1109/ICIP46576.2022.9897990](http://dx.doi.org/10.1109/ICIP46576.2022.9897990)
- **[Bät+26]** A. Bätz, P. Klasek, S.-Y. Ham, P. Neumaier, M. Köppel, M. Lauer. *Railway Artificial Intelligence Learning Benchmark (RAIL-BENCH): A Benchmark Suite for Perception in the Railway Domain*, 2026. arXiv: [2604.22507](https://arxiv.org/abs/2604.22507)
- **[Ins26]** Institute for Measurement and Control Systems, KIT. *RAIL-BENCH project presentation*, kickoff presentation, April 2026. [Slides](https://www.mrt.kit.edu/railbench_data/additional_information/railbench_project_presentation.pdf)
- **[Joc+26]** G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, M. E. Kalfaoglu. *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, 2026. arXiv: [2606.03748](https://arxiv.org/abs/2606.03748)
- **[Kha+24]** R. Khanam, M. Hussain. *YOLOv11: An Overview of the Key Architectural Enhancements*, 2024. arXiv: [2410.17725](https://arxiv.org/abs/2410.17725)
- **[Liu+26]** L. Liu, Y. Hou, Y. Li, Q. Wang, Y. Sha, Y. Yu, Y. Wang, P. Ru, X. Yu, X. Shen. *EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation*. Transactions on Machine Learning Research, 2026.
- **[oMea+]** Institute of Measurement and Control Systems, KIT. *RAIL-BENCH Object Detection*. [https://www.mrt.kit.edu/railbench/object_detection](https://www.mrt.kit.edu/railbench/object_detection)
- **[Pad+21]** R. Padilla, W. L. Passos, T. L. B. Dias, S. L. Netto, E. A. B. da Silva. *A comparative analysis of object detection metrics with a companion open-source toolkit*. Electronics, 10(3):279, 2021. DOI: [10.3390/electronics10030279](https://doi.org/10.3390/electronics10030279)
- **[Ult26]** Ultralytics. *Ultralytics YOLO11 documentation*, 2026. [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)

---

## Citation

If you refer to this work, please cite the project report:

```bibtex
@misc{eric_toudik_railbench,
  author       = {Walsh, Eric and Isavi Arktan, Toudik},
  title        = {{RAIL-BENCH} Object Detection Challenge: Project Report},
  year         = {2026},
  howpublished = {Project report, Chair of Pattern Recognition,
                  Friedrich-Alexander-University Erlangen-Nuremberg},
  note         = {Advisor: Dr.-Ing. Vincent Christlein}
}
```
