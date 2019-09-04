import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from Janghoo_Model.modeltype import AlexNet, AlexNet_original, SmallNet

# output class 개수는 4개여야만 합니다!! 아직 안만들었음!!


from keras.models import load_model


train_directory = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/maplestory_map_source/train'
valid_directory = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/maplestory_map_source/valid'
test_directory = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/maplestory_map_source/test'
checkpoint_path = "trainingcheckpoint/cp.ckpt"


train_batch_size = 5
valid_batch_size = 5
test_batch_size = 5
epochs = 10




train_directory_walk = (os.walk(train_directory))
valid_directory_walk = (os.walk(valid_directory))
test_directory_walk = (os.walk(test_directory))
trainset_count = 0
for base, dirs, names in train_directory_walk :
#    print(base, dirs, names)
    trainset_count = trainset_count + len(names)
print("count : there is ", trainset_count, "files in the path")
testset_count = 0
for base, dirs, names in test_directory_walk :
#    print(base, dirs, names)
    testset_count = testset_count + len(names)
print("count : there is ", testset_count, "files in the path")
validset_count = 0
for base, dirs, names in valid_directory_walk :
#    print(base, dirs, names)
    valid_count = testset_count + len(names)
print("count : there is ", validset_count, "files in the path")



# 먼저 ImageDataGenerator 클래스를 이용하여 객체를 생성한 뒤
# flow_from_directory() 함수를 호출하여 제네레이터(generator)를 생성합니다.
# flow_from_directory() 함수의 주요인자는 다음과 같습니다.
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size= (80, 80),
    batch_size = train_batch_size,
    class_mode = 'categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size= (80, 80),
    batch_size = test_batch_size,
    class_mode = 'categorical'
)

valid_generator = ImageDataGenerator(rescale=1./255)
valid_generator = valid_generator.flow_from_directory(
    valid_directory,
    target_size= (80, 80),
    batch_size = valid_batch_size,
    class_mode = 'categorical'
)



# model 을 선택합니다.
# model = SmallNet()
model = SmallNet()


# model 을 컴파일합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()



# 체크포인트 콜백 만들기
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# 모델 학습, 학습 기록 저장
history = \
    model.fit_generator(
        train_generator,                #훈련데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 train_generator으로 지정합니다.
#        steps_per_epoch=trainset_count/train_batch_size,             #한 epoch에 사용한 스텝 수를 지정합니다. 총 n개의 데이터가 있고, batch size 를 지정했으니 그냥 나눈값을 투입.
        epochs=epochs,
        validation_data=valid_generator, #검증데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 validation_generator으로 지정합니다.
        # 원래 validation generator 을 사용해야 하지만 우선 그냥 이걸로 할래요..
        callbacks = [cp_callback]
#        validation_steps=testset_count/train_batch_size
    )             #한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 홍 15개의 검증 샘플이 있고 배치사이즈가 3이므로 5 스텝으로 지정합니다.



# 모델 평가
model.save('Janghoo_model.h5')
loss, acc = model.evaluate_generator(test_generator, steps=5)
print("\n test set >> loss : {} , Acc : {} ". format(loss, acc))



# 모델 기록 시각화
from Janghoo_Model.visuallize import plt_show_acc, plt_show_loss
plt_show_loss(history)
plt_show_acc(history)