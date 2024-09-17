# ISIC 2024 - Skin Cancer Detection with 3D-TBP

## Goal

### <a href ="https://www.kaggle.com/competitions/isic-2024-challenge">ISIC 2024 - Skin Cancer Detection with 3D-TBP </a> **Top 40%**

Identify cancers among skin lesions cropped from 3D total body photographs.

In this competition, you'll develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). 

The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes. 

Bbinary classification algorithm could be used in settings without access to specialized care and improve triage for early skin cancer detection.

3D TBD에서 잘라낸 피부 병변 중 암을 식별하는 것이 목표이다. 

3D total body photos (TBP)에서 single-lesion crops을 사용해 조직학적으로 확인된 피부암 사례를 식별하는 image-based algorithms을 개발한다. 

## Definition

### Image + Tabular Dataset

대규모 Image, Tabular Data를 결합해 두 Data의 장점을 극대화한 예측 Model을 개발한다. 

LightGBM Model을 기준으로 Image에서 추출한 예측 결과(해당 병변이 악성일 확률, 0-1)와 다양한 환자 Metadata를 효과적으로 통합해 Data의 다양성과 복잡성을 반영한다. 

Image와 Metadata의 상호보완적 특징을 활용해 더 나은 예측 성능을 확보한다. 

3D-TBP Image와 Tabular Data를 결합해 피부암 진단에 유의미한 Feature를 추출하고 활용한다. 

### Analysis of Tabular Dataset

Missing Values, Categorical Variables One-Hot Encoding, Numerical Variables Normalization, Feature Engineering을 통해 새로운 Variables를 생성한다. 

기존 Numerical Variables를 조합해 의미 있는 Variables를 추가하고, 환자 Group 내에서 변수를 Normalization 하는 등 Data를 세밀하게 다듬어 Model 성능 향상에 기여한다. 

### LightGBM Modeling

LightGBM Model을 주요 예측 도구로 사용해 다양한 Hyperparameter를 Setting, Optimization해 성능을 극대화 한다. 

StratifiedGroupKFold 교차 검증을 통해 Model의 일반화 성능을 평가한다. 

Data의 Class 불균형 문제를 해결하기 위해 Oversampling 및 Undersampling 전략을 사용해 Model의 성능을 개선한다.

## Overview

### **Dataset**

**Directory Structure**

```
├── /kaggle/working
└── /kaggle/input/isic-2024-challenge/train-image
    ├── train-image/
    │   └── image/
    │       ├── ISIC_0015670.jpg
    │       ├── ISIC_0015845.jpg
    │       └── ...
    │
    ├── test-image/
    │   └── image/
    │       └── ...
    │
    ├── sample_submission.csv
    ├── test-image.hdf5
    ├── test-metadata.csv
    ├── train-image.hdf5
    └── train-metadata.csv

```

**File**
> train-image/ : Image files for the training set(Provided for train only). <br>
> train-image.hdf5 : Training image data contained in a single hdf5 file, with the isic_id as key. <br>
> train-metadata.csv : Metadata for the training set. <br>
> test-image.hdf5 : Test image data contained in a single hdf5 file, with the isic_id as key. <br>
> test-metadata.csv : Metadata for the test subset. <br>
> sample_submission.csv : A sample submission file in the correct format. <br>

**Matadata** <br>
> lesion_id : 관심 병변에 대한 고유 식별자이다. 병변이 중요하다 판단된 경우 수동으로 Tag 된다. <br>
> iddx_full ~ iddx_5 : 병변에 대한 세부적인 진단 정보이다. iddx_full은 완전히 분류된 진단 정보, iddx_1부터 iddx_5까지 점진적인 세부 진단을 나타낸다. <br>
> mel_thick_mm : 흑색종의 깊이를 나타내며 종양의 진행 정도를 평가한다. <br>
> tbp_lv_dnn_lesion_confidence : 병변 확신도 점수, 0-100 사이의 값을 가지며 병변이 악성일 가능성을 나타낸다.  

**`train-metadata.csv`**

| Field Name | Description |
| :--- | :--- |
| target | Binary class {0: benign, 1: malignant}. |
| lesion_id | Unique lesion identifier. Present in lesions that were manually tagged as a lesion of interest. 병변의 고유 식별자. |
| iddx_full | Fully classified lesion diagnosis. 병변 전체의 진단. iddx_1-iddx_5를 :: 구분자로 합쳐놓은 상태. |
| iddx_1 | First level lesion diagnosis. 1차 진단 범주. |
| iddx_2 | Second level lesion diagnosis. 2차 진단 범주. |
| iddx_3 | Third level lesion diagnosis. 3차 진단 범주. |
| iddx_4 | Fourth level lesion diagnosis. 4차 진단 범주. |
| iddx_5 | Fifth level lesion diagnosis. 5차 진단 범주. |
| mel_mitotic_index | Mitotic index of invasive malignant melanomas. 침윤성 악성 흑색종의 유사 분열 지수. |
| mel_thick_mm | Thickness in depth of melanoma invasion. 흑색종 침윤 두께. |
| tbp_lv_dnn_lesion_confidence | Lesion confidence score (0-100 scale). + 병변 NN 신뢰도 점수. |

**`train-metadata.csv and test-metadata.csv`**

| Field Name | Description |
| :--- | :--- |
| isic_id | Unique case identifier. 각 Sample의 고유 식별자. |
| patient_id | Unique patient identifier. 환자의 고유 식별자. |
| age_approx | Approximate age of patient at time of imaging. 촬영 당시 환자의 대략적인 나이. |
| sex | Sex of the person. |
| anatom_site_general | Location of the lesion on the patient's body. 병변이 위치한 신체 부위. |
| clin_size_long_diam_mm | Maximum diameter of the lesion (mm). + 병변의 최대 직경. |
| image_type | Structured field of the ISIC Archive for image type. 이미지 유형(구조화 된 Field). |
| tbp_tile_type | Lighting modality of the 3D TBP source image. 3D TBP 원본 Image의 조명 방식 |
| tbp_lv_A | A inside lesion. + 병변 내부의 A 값. |
| tbp_lv_Aex | A outside lesion. + 병변 외부의 A 값. |
| tbp_lv_B | B inside lesion. + 병변 내부의 B 값. |
| tbp_lv_Bext | B outside lesion.+ 병변 외부의 B 값. |
| tbp_lv_C | Chroma inside lesion.+ 병변 내부의 색도. |
| tbp_lv_Cext | Chroma outside lesion.+ 병변 외부의 색도. |
| tbp_lv_H | Hue inside the lesion, calculated as the angle of A* and B* in L*A*B* color space.  Typical values range from 25 (red) to 75 (brown). + 병변 내부의 색상. |
| tbp_lv_Hext | Hue outside lesion. + 병변 외부의 색상. |
| tbp_lv_L | L inside lesion. + 병변 내부의 명도. |
| tbp_lv_Lext | L outside lesion. + 병변 외부의 명도. |
| tbp_lv_areaMM2 | 병변의 면적. |
| tbp_lv_area_perim_ratio | Border jaggedness, the ratio between lesions perimeter and area.  Circular lesions will have low values, irregular shaped lesions will have higher values. Values range 0-10. + 병변 경계의 울퉁불퉁함의 비율. |
| tbp_lv_color_std_mean | Color irregularity, calculated as the variance of colors within the lesion's boundary. 병변 경계 내 색상 불규칙성. |
| tbp_lv_deltaA | Average A contrast (inside vs. outside lesion). + 병변 내 외부의 A 대비값. |
| tbp_lv_deltaB | Average B contrast (inside vs. outside lesion). + 병변 내 외부의 B 대비값. |
| tbp_lv_deltaL | Average L contrast (inside vs. outside lesion). + 병변 내 외부의 명도 대비값. |
| tbp_lv_deltaLB | 병변의 명도 및 색상 대비값. |
| tbp_lv_deltaLBnorm | Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles, high contrast lesions tend to be those with darker pigment. Calculated as the average delta L*B* of the lesion relative to its immediate background in L*A*B* color space. Typical values range from 5.5 to 25. + 병변과 주변 피부의 명도 및 색상 대비. |
| tbp_lv_eccentricity | Eccentricity. + 병변의 이심률. |
| tbp_lv_location | Classification of anatomical location, divides arms & legs to upper & lower, torso into thirds. + 병변의 해부학적 위치(팔, 다리, 몸통 등으로 구분). |
| tbp_lv_location_simple | Classification of anatomical location, simple. + 병변의 간단한 해부학적 위치. |
| tbp_lv_minorAxisMM | Smallest lesion diameter (mm). + 병변의 최소 직경. |
| tbp_lv_nevi_confidence | Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus.  The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist. +, ++ 병변이 모반일 가능성에 대한 NN 에측 확률. |
| tbp_lv_norm_border | Border irregularity (0-10 scale), the normalized average of border jaggedness and asymmetry. + 경계 불규칙성 |
| tbp_lv_norm_color | Color variation (0-10 scale), the normalized average of color asymmetry and color irregularity. + 색상 변이(정규화 된 값). |
| tbp_lv_perimeterMM | Perimeter of lesion (mm). + 병변의 둘레. |
| tbp_lv_radial_color_std_max | Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in L*A*B* color space within concentric rings originating from the lesion center. Values range 0-10. + 병변 내 색상의 비대칭성 |
| tbp_lv_stdL | Standard deviation of L inside lesion. + 병변 내부 명도의 표준 편차. |
| tbp_lv_Lext | Standard deviation of L outside lesion. + 병변 외부 명도의 표준 편차. |
| tbp_lv_symm_2axis | Border asymmetry, a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis.  Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation.  The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10. + 병변 경계 비대칭성. |
| tbp_lv_symm_2axis_angle | Lesion border asymmetry angle. + 병변 경계 비대칭성 각도. |
| tbp_lv_x | X-coordinate of the lesion on 3D TBP. + 3D TBP에서 병변의 X 좌표. |
| tbp_lv_y | Y-coordinate of the lesion on 3D TBP. + 3D TBP에서 병변의 Y 좌표. |
| tbp_lv_z | Z-coordinate of the lesion on 3D TBP. + 3D TBP에서 병변의 Z 좌표. |
| attribution | Image attribution, synonymous with image source. |
| copyright_license | Copyright license. |


### **Evaluation**

**Primary Scoring Metric**

Submissions are evaluated on [partial area under the ROC curve (pAUC)](https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve) above 80% true positive rate (TPR) for binary classification of malignant examples. (See the implementation in the notebook [ISIC pAUC-aboveTPR](https://www.kaggle.com/code/metric/isic-pauc-abovetpr).)

pAUC는 ROC 곡선의 특정 부분, 즉 TPR(참 양성 비율)이 80% 이상인 영역을 평가한다. 

이는 높은 민감도를 요구하는 실제 임상 환경을 반영하기 위함이다. 

점수는 [0.0, 0.2] 범위 내에서 주어지며, TPR 80% 이상에서의 분류 성능을 강조한다. Image에서 파란색과 빨간색 영역이 각각 두 Algorithm의 pAUC를 나타낸다. 

<img width="800" height="400" src="https://github.com/user-attachments/assets/ae199c45-0137-4efd-ad6d-a48757546501">

### **Submission**

For each image (`isic_id`) in the test set, you must predict the probability (`target`) that the lesion is **malignant**. The file should contain a header and have the following format:

Test Set에 포함된 Image(’isic_id’)에 대해 **malignant**(악성)일 확률(’target’)을 예측해야 한다. 

Submission File에는 Header가 포함되어야 하며, 아래와 같은 형식이다. 

```python
isic_id,target
ISIC_0015657,0.7
ISIC_0015729,0.9
ISIC_0015740,0.8
etc.
```

### **Timeline**

- **June 26, 2024** - Start Date.
- **August 30, 2024** - Entry Deadline. You must accept the competition rules before this date in order to compete.
- **August 30, 2024** - Team Merger Deadline. This is the last day participants may join or merge teams.
- **September 6, 2024** - Final Submission Deadline.
- **September 20, 2024** - Deadline for potential prize-winners to publish their solution code and write-ups.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Technical Stack

`Python` `TensorFlow` `PyTorch`

## Model Architecture

<img width ="400" height="2800" src="https://github.com/user-attachments/assets/03d6f83a-3088-4e8d-bdb3-0f025a24bbf8">

## Result

### Kaggle Score

<img width="975" alt="스크린샷 2024-09-10 오후 2 40 40" src="https://github.com/user-attachments/assets/ebfbcf37-461e-4aa2-b10a-eeb26461cb12">

### Team Project Report

<a href="https://www.notion.so/ISIC-2024-Skin-Cancer-Detection-with-3D-TBP-245e61e7b96e4f4c8054988242b797d8" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/baaa10f4-56b7-4b1d-8740-ec61aa433e13" width="80" height="80" alt="Portfolio"></a>

> This is the Team Project Report, Click the icon to move it.
