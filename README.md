# MPFM-S3D

MPFM-S3D is a two-stage detector for strip steel surface defect detection.

### trained model

We provide pth of our MRANet trained on [GC10-Det]() dataset: [GC10.pth](https://pan.baidu.com/s/1NaqebHmVrWv8A3PPgP0FgQ?pwd=a4aj) (Code: a4aj), and on the [Neu-Det]() dataset: [NEU.pth](https://pan.baidu.com/s/13QXfs_MtPXcIHRldjrf8SA?pwd=xyty) (Code: xyty).  

### training and validation

Our MPFM-S3D is implemented based on the [mmdetection-3.3.0](https://github.com/open-mmlab/mmdetection)

- **Training (using GC10 dataset as an example)**:  
  
  ```bash
  python train.py configs/mpfm_s3d/lgnet_rbb_rab_mssab_pcsab_mffn_36e_GC10-COCO-640.py
  ```

- **validation (using GC10 dataset as an example)**:  
  
  ```bash
  python train.py configs/mpfm_s3d/lgnet_rbb_rab_mssab_pcsab_mffn_36e_GC10-COCO-640.py GC10.pth 
  ```