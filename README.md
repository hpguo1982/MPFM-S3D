# MPFM-S3D

MPFM-S3D is a two-stage detector for strip steel surface defect detection.

### trained model

We provide pth of our MRANet trained on [GC10-Det]() dataset: , and on the [Neu-Det]() dataset: 

### training and validation

Our MPFM-S3D is implemented based on the [mmdetection-3.30](https://github.com/open-mmlab/mmdetection)

- **Training (using GC10 dataset as an example)**:  
  
  ```bash
  python train.py configs/mpfm_s3d/lgnet_rbb_rab_mssab_pcsab_mffn_36e_GC10-COCO-640.py
  ```

- **validation (using GC10 dataset as an example)**:  
  
  ```bash
  python train.py configs/mpfm_s3d/lgnet_rbb_rab_mssab_pcsab_mffn_36e_GC10-COCO-640.py your_trained_model_path
  ```