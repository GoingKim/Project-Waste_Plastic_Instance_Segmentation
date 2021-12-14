# Waste-plastic instance segmentation

![Alt text](./samples/detectors.png?raw=true "Optional Title")

이 README.md 파일은 폐플라스틱 객체 검출 예측 경진대회에 제출하는 소스코드 설명합니다.
이번 경진대회에서는 총 4개의 클래스를 instance segmentation을 통해 검출해야 합니다.
소스코드를 재생산하기 위한 자세한 설명을 위해 아래를 읽어보시기 바랍니다.


## Summary
* DetectoRS (HTC + ResNet-50), Mask R-CNN, yolact
* Light augmentation 
* Test data training

## OS environment
- Ryzen 7 3700x
- GeForce RTX 3090 

## Python packages
- Ubuntu 18.04 LTS
- CUDA 11.1
- CuDNN 8.0.5
- Python 3.9.7
- requirements.txt
```
$ defpack="ipykernel"
$ conda create --name seg_p3.9 python=3.9 $defpack
$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
$ pip install -r requirements.txt
```

anaconda yml 파일도 업로드 해드리오나 torch==1.8.0+cu111 버전 미배포로 위 방식의 설치를 권장합니다.

## Dataset
- 4 of classes, ('pet', 'ps', 'pp', 'pe')
- 2024 X 2024 / 2048 X 2048, jpg image
- Train data has 1000 files for each class
- Test data has 100 files for each class

./mmdetection/data folder structure should be:
```
data
├── final.json
├── train.json
├── val.json
├── anno_final
│   ├── PET_002_143.json
│   ├── ...
├── anno_test
│   ├── PET_002_1241.json
│   ├── ...
├── anno_train
│   ├── PET_002_143.json
│   ├── ...
├── final
│   ├── PET_002_143.jpg
│   ├── ...
├── test
│   ├── PET_002_1241.jpg
│   ├── ...
├── train
│   ├── PET_002_143.jpg
│   ├── ...
```

## Pre-requirements
updated mmdetection를 사용하기 위한 몇 가지 설정을 알려드립니다. 
(제공한 소스에는 적용되어 있어 설정할 필요 없음)

- cfg.data cofig 내부 train/val/test value값으로 classes를 추가해야 합니다. 
  cfg.data = dict(... train=dict(..., classes=(..., ..., ...) ,...), ...)
- train_cfg, test_cfg에 nms=dict(type='nms', iou_threshold=0.7) 있는지 확인해야 합니다. 
- train_cfg, test_cfg에 max_num이 있는 경우 max_per_img로 변경해야 합니다.
- ./mmdetection/mmdet/dataset/coco.py 와 ./mmdetection/mmdet/core/evaluation/class_name.py의 
  class를 ('pet', 'ps', 'pp', 'pe')로 변경해야 합니다.
- DetectoRS 모델을 사용할 경우 ./mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py 파일에서
  '../htc/htc_r50_fpn_1x_coco.py'를 '../htc/htc_without_semantic_r50_fpn_1x_coco.py'으로 변경해야 합니다.

## Model
* 모델 1 : [DetectoRS (HTC + ResNet50)](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors)
* 모델 2 : [Mask R-CNN (ResNet50-FPN)](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn)
* 모델 3 : [yolact (ResNet101-FPN)](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolact)
* Optimizer : SGD with initial LR 0.01 (but, yolact has 0.001)

Instance segmentation을 수행하기 위해서 3가지 모델을 선정하여 비교하였습니다.
본 대회는 Inference 시 속도(Fps)도 중요하기 때문에 속도에서 좋은 성능을 보이는 yolact와
가장 대중적이고 잘 알려진 Mask R-CNN, 그리고 20년 SOTA로 성능, 속도 둘 다 뛰어난 DetectoRS을 사용하였습니다.
해당 링크를 따라가면 Pre-trained model을 다운받을 수 있습니다. 
다운받은 모델은 ./mmdetection/checkpoints 폴더에 저장해야 합니다.

## Train & Test model

### Train
 
1. submission_coco_formating.ipynb 파일을 열어서 실행시킨 후 train.json 파일을 생성합니다.
2. submission_eda_and_preprocessing.ipynb 파일을 열어서 실행시킵니다.
   (preprocessing 과정에서 직접 파일을 찾아 삭제해야 하나 소스에는 적용되어 있음)
3. 학습을 원하는 모델 파일을 열어서(ex> submission_train_detectors.ipynb) 실행시켜 학습을 진행합니다.
4. log 폴더에서 생성된 checkpoint를 확인합니다.

### Test
1. submission_coco_formating.ipynb 파일을 열어서 실행시킨 후 test.json 파일을 생성합니다.
2. dectors_config.py 파일을 열어서 config.data의 test파일 경로를 수정합니다.
3. submission_inference.ipynb를 실행합니다.

./mmdetection folder structure should be:
```
mmdetection
├── submission_coco_formating.ipynb
├── submission_eda.ipynb
├── submission_inference.ipynb
├── submission_train_detectors.ipynb
├── submission_train_maskrcnn.ipynb
├── submission_train_yolact.ipynb
├── detectors_config.py
├── ... etc ... (mmdetection default files) 
├── checkpoints
│   ├── detectors_htc_r50_1x_coco-329b1453.pth
│   ├── ...
├── data
│   ├── final.json
│   ├── ...
├── log
├── ... etc ... (mmdetection default folders)
```

## Performance & Model selection
| Network                  | image-size | Fps (task/s) | Valid bbox_mAP_50 | Valid segm_mAP_50 |
|:-------------------------| :----------|:-------------|:------------------|:------------------|
| DetectoRS                | 768        | 5.9          | 0.997             | 0.997             |
| DetectoRS (with soft-nms)| 768        |              | 0.996             | 0.996             |
| DetectoRS                | 1333x768   |              | 0.996             | 0.996             |
| Mask R-CNN               | 1024       | 13.2         | 0.99              | 0.99              |
| yolact                   | 768        | 14.0         | 0.992             | 0.992             |

Valid bbox_mAP 기준으로 대체적으로 모두 성능이 좋은 편입니다.
하지만, Fps 기준으로 속도를 고려했을 때 yloact가 다른 모델에 비해 속도가 훨씬 빠른 것으로 알 수 있습니다.
