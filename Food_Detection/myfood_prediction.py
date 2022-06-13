import os
import cv2
import random
import numpy as np
from tensorflow import keras
from glob import glob
from PIL import Image


base_dir = '..\\Food_Detection\\Myfood\\images\\'
test_images = glob(os.path.join(base_dir, 'test_set\\jajangmyeon\\*.jpg'))
pasta_img = '../Food_Detection/Myfood/images/test_set/bibimbap/83215.jpg'
cnn_model = keras.models.load_model("./myfood_conv3_tf1.h5")
class_names = ['apple', 'bibimbap', 'bulgogi', 'chocolate_cake', 'fried_egg', 'hamburger', 'jajangmyeon', 'kimbap', 'kimchi_stew', 'pizza', 'ramen', 'sandwich', 'steak', 'sushi']

randIdx = np.random.randint(0, 149)
random_img = cv2.imread(test_images[randIdx], cv2.IMREAD_COLOR)
# random_img = cv2.imread(pasta_img, cv2.IMREAD_COLOR)

# resize 방법1(224*224로 resize 한 경우)
img_resize = np.resize(random_img, (1, 224, 224, 3))

# 정답 출력
route = test_images[randIdx].split("\\")
answer = route[5]

predictions = cnn_model.predict(img_resize)
pred_food = np.argmax(predictions)

print(test_images[randIdx])
print(predictions)
print("음식 정답 : {}".format(answer))
print("예측한 음식 : {}, {}".format(pred_food , class_names[pred_food]))

# image 출력(224*224로 resize 안한 경우)
small_img = cv2.resize(random_img, (512, 512))
cv2.imshow("resize img", small_img)
cv2.waitKey()


# for i in range(len(test_images)):
#     # randIdx = np.random.randint(0, 149)
#     random_img = cv2.imread(test_images[i], cv2.IMREAD_COLOR)
#     # random_img = cv2.imread(pasta_img, cv2.IMREAD_COLOR)
#
#     # resize 방법1(224*224로 resize 한 경우)
#     img_resize = np.resize(random_img, (1, 224, 224, 3))
#
#     # 정답 출력
#     route = test_images[i].split("\\")
#     answer = route[5]
#
#     predictions = cnn_model.predict(img_resize)
#     pred_food = np.argmax(predictions)
#     # print("음식 정답 : {}/ 예측한 음식 : {}".format(answer, class_names[pred_food]))
#     print("예측한 음식 : {}/ 사진명 : {}".format(class_names[pred_food], route[6]))
#
#     if(answer == class_names[pred_food]):
#         small_img = cv2.resize(random_img, (512, 512))
#         cv2.imshow("resize img", small_img)
#         cv2.waitKey()
#
#         break
