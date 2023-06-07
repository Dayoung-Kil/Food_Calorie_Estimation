## Food Calorie Estimation
#### 2022-1 기계 학습과 인공지능 Term Project - 김준용, 길다영, 장재훈

<br>

### 1. 목표 & 기대효과
#### 1-1. 목표
- 음식 이미지 인식 후, 음식의 칼로리 추정 


#### 1-2. 기대 효과
- 일일이 음식의 종류와 양을 기입하는 불편함 없이, 사진 한장으로 칼로리 계산 가능.
-  부피를 계산하여 칼로리를 계산하기 때문에 보다 정확함.

<br>

### 2. Work Flow
- Input Img → Food Detection → Volume Estimation → Calorie Calculation → Results

![image](https://user-images.githubusercontent.com/53934639/173486316-41400309-4041-429d-b04d-02154abc8a7b.png)

<br>

### 3. 방법
#### 3-1. Food Detection
- 3개의 Convolution layer와 Pooling layer로 구성
- Keras를 통한 이미지 로드 및 전처리 방법 : image.ImageDataGenerator
- atch_size는 150, epochs는 100으로 설정
- 결과 Loss: 108.18%, ACC : 73.42%가 나옴

#### 👩‍💻 Food detection만 따로 실행하는 방법
- [myfood_prediction.py](https://github.com/arittung/Food_Calorie_Estimation/blob/main/Food_Detection/myfood_prediction.py) 실행

<br>

#### 3-2. Volume Estimation
- [AlexGraikos/food_volume_estimation](https://github.com/AlexGraikos/food_volume_estimation) 코드 이용함

![image](https://user-images.githubusercontent.com/53934639/173487815-de4c02df-a99b-4056-9dd7-5269058b4178.png)

<br>


#### 3-3. [Calorie Estimation](https://github.com/arittung/Food_Calorie_Estimation/blob/main/Food_volume_estimation/volume_estimator.py#L510)
- Food 101 데이터 셋의 음식 종류와 g수에 따른 칼로리, 영양성분을 나타내기 위해 dict = {'food' : [g, kcal, 탄수화물, 단백질, 지방, 당류]} 형태의 딕셔너리 생성
- (추정한 volume / g) * kcal 로 input image의 최종 칼로리 계산

![image](https://user-images.githubusercontent.com/53934639/173488131-00b159fb-d0f5-49e6-8468-f822d9020bb7.png)



<br>

### 4. Code 실행 방법
#### 4-1 환경 설정 
- requirements.txt 설치
```
pip install -r requirements.txt 
```
- [food dataset](https://drive.google.com/file/d/1PE3r-ve0FOOMEwIzZJfEo5PXH6KtDZOF/view?usp=sharing) 다운로드 후 ./Food_Detection/ 에 압축 풀기
- [CNN Model](https://drive.google.com/file/d/1B_aWg1_1JIbCU6cbNcqzwHqu7AY0gZ4M/view?usp=sharing) 다운로드 후 ./Food_Detection/ 에 압축 풀기

<br>

#### 4-2 [volume_estimator.py](https://github.com/arittung/Food_Calorie_Estimation/blob/main/Food_volume_estimation/volume_estimator.py) 파일 실행. 
  - food_detection, volume estimation, calorie calculation이 한번에 실행되도록 구현

```
python volume_estimator.py --input_images ../Food_Detection/Myfood/images/test_set/kimbap/Img_069_0755.jpg --depth_model_architecture depth_architecture.json --depth_model_weights depth_weights.h5 --segmentation_weights segmentation_weights.h5
```

##### 👩‍💻 code 설명
- --input_images : input image 경로
- depth_model_architecture, depth_model_weights, segmentation_weights는 [여기](https://github.com/AlexGraikos/food_volume_estimation#models)에서 다운로드 후 ./Food_volume_estimation/ 에 넣기
- --plot_results : depth와 object mask, plate contour을 그림으로 나타냄
- --plate_diameter_prior : 접시 지름

<br>

### 5. 실행 결과
- 음식 종류와 g수에 따른 칼로리, 영양성분 나타남.

![image](https://user-images.githubusercontent.com/53934639/173487955-0bf8e1a4-d5cc-4032-aeb8-fe60ecab5dfc.png)


