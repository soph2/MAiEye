import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os



# 이 함수는
# 학습 데이터셋, 검증 데이터셋, 테스트 데이터셋이 지정된 폴더에서

def trainWithCheckpoint() :
    pass




def callModel() :
    if __name__ == "__main__" :
        from modeltype import AlexNet, AlexNet_original, SmallNet
        from visuallize import plt_show_acc, plt_show_loss
    else :
        from .modeltype import  AlexNet, AlexNet_original, SmallNet
        from .visuallize import plt_show_acc, plt_show_loss



    # output class 개수는 4개여야만 합니다!! 다양한 크기를 받을 수 있도록 만들지는 않았습니다!!
    train_directory = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/maplestory_map_source/train'
    valid_directory = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/maplestory_map_source/valid'
    test_directory  = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/maplestory_map_source/test'
    checkpoint_path = "trainingcheckpoint/cp.ckpt"


    train_batch_size = 5
    valid_batch_size = 5
    test_batch_size  = 5
    epochs = 10



    # 각 폴더 안에 들어있는 파일 갯수 세기
    train_directory_walk = (os.walk(train_directory))
    valid_directory_walk = (os.walk(valid_directory))
    test_directory_walk = (os.walk(test_directory))
    trainset_count = 0
    for base, dirs, names in train_directory_walk :
        trainset_count = trainset_count + len(names)
    print("count : there is ", trainset_count, "files in the path")
    testset_count = 0
    for base, dirs, names in test_directory_walk :
        testset_count = testset_count + len(names)
    print("count : there is ", testset_count, "files in the path")
    validset_count = 0
    for base, dirs, names in valid_directory_walk :
        valid_count = testset_count + len(names)
    print("count : there is ", validset_count, "files in the path")



    # 먼저 ImageDataGenerator 클래스를 이용하여 객체를 생성한 뒤
    # flow_from_directory() 함수를 호출하여 제네레이터(generator)를 생성합니다.
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


    # Tensorboard 를 위한 log 디렉토리 만들기
    tensorboard_logpath = 'tensorboardlog'
    if not os.path.isdir(tensorboard_logpath) :
        # save folder 이 존재하지 않는다면..
        os.makedirs(tensorboard_logpath)
        # 폴더를 생성한다.


    # 체크포인트 콜백 만들기
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logpath,
                                                          histogram_freq=1)
    # 모델 학습, 학습 기록 저장
    history = \
        model.fit_generator(
            train_data = train_generator,
    # 훈련데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 train_generator으로 지정합니다.

            validation_data=valid_generator,
    # 검증데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 validation_generator으로 지정합니다.

            epochs=epochs,
            callbacks = [cp_callback]
        )


    # 모델 평가
    model.save('Janghoo_model.h5')
    loss, acc = model.evaluate_generator(test_generator, steps=5)
    print("\n test set >> loss : {} , Acc : {} ". format(loss, acc))


    # 모델 기록 시각화
    plt_show_loss(history)
    plt_show_acc(history)


if __name__ == "__main__" :
    print("\n\n call module separately \n\n")