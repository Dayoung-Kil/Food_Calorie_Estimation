import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# base_dir은 현재 myfood가 들어있는 폴더로 지정해줍니다.
base_dir = './myfood/images/'
train_images = glob(os.path.join(base_dir, 'train_set/*/*.jpg'))
test_images = glob(os.path.join(base_dir, 'test_set/*/*.jpg'))

# 데이터 수 출력
print(f'train_images 수 : {len(train_images)}')
print(f'test_images 수 : {len(test_images)}\n')


# # 데이터 사이즈 변경
# for i in range(len(train_images)):
#     train_image1 = Image.open(train_images[i])
#     train_image_size = train_image1.resize((224, 224))
#
# for i in range(len(test_images)):
#     test_image1 = Image.open(test_images[i])
#     test_image_size = test_image1.resize((224, 224))
#
# print(f'train_image_size : {train_image_size}')


# 클래스 출력
train_classes = os.listdir(os.path.join(base_dir, 'train_set'))
test_classes = os.listdir(os.path.join(base_dir, 'test_set'))
num_classes = len(train_classes)
print(f'classes 수 : {num_classes}')

# 클래스별 train, test 데이터 수
for train_class_name in train_classes:
    train_class_dir = glob(os.path.join(base_dir, 'train_set', train_class_name, '*.jpg'))
    print(f'{train_class_name} 수 : {len(train_class_dir)}')
print("--------------")
for test_class_name in test_classes:
    test_class_dir = glob(os.path.join(base_dir, 'test_set', test_class_name, '*.jpg'))
    print(f'{test_class_name} 수 : {len(test_class_dir)}')


train_dir = os.path.join(base_dir, 'train_set')
val_dir = os.path.join(base_dir, 'test_set')

# 제너레이터 정의
train_argmentation = ImageDataGenerator(
    rescale=1./255, # 이미지 0-1 사이의 값으로 변경
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

val_argmentation = ImageDataGenerator(
    rescale=1./255
)

augment_size = 150
train_generator = train_argmentation.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=augment_size,
    shuffle=True,
    class_mode='categorical'
)
val_generator = val_argmentation.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=augment_size,
    shuffle=False,
    class_mode='categorical'
)


cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(224, 224, 3), kernel_size=(3,3), filters=32, strides=1, padding='same', activation='relu'),
    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', strides=1, activation='relu'),
    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=512, padding='same', strides=1, activation='relu'),
    # tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    #
    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=512, padding='same', strides=1, activation='relu'),
    # tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# 학습률 저장
cnn_history = cnn_model.fit(train_generator, validation_data=val_generator, epochs=100)

# 손실, 정확도 계산
loss, accuracy = cnn_model.evaluate(val_generator)
print('Loss: {:.2f}%, ACC : {:.2f}%'.format(loss*100, accuracy*100))


# 모델 저장
cnn_model.save("myfood_conv3_tf1.h5")


# 학습 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['loss'], 'b-', label='Train_loss', marker='o')
plt.plot(cnn_history.history['val_loss'], 'g-', label='Test_loss', marker='o')
plt.title('Food101 Model Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], 'b-', label='Train_Accuracy', marker='o')
plt.plot(cnn_history.history['val_accuracy'], 'g-', label='Test_Accuracy', marker='o')
plt.title('Food101 Model Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()