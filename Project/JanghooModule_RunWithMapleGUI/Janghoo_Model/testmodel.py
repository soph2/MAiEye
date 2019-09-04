import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


def load_model() :
    test_batch_size = 5
    test_directory = 'C:/Users/user/Desktop/programming_PROJECTS/MAiEye/Data/test'

    # 모델 불러오기
    model = tf.keras.models.load_model('./Janghoo_model.h5')


    #from keras.models import load_model
    #model = load_model('Janghoo_model.h5')
    #SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))



    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_directory,
        target_size= (80, 80),
        batch_size = test_batch_size,
        class_mode = 'categorical'
    )

    
    # 모델 사용하기
    # 클래스에 해당하는 열을 알기 위해서는 제네레이터의 ‘class_indices’를 출력하면 해당 열의 클래스명을 알려줍니다.
    # 모델 사용 시에 제네레이터에서 제공되는 샘플을 입력할 때는 predict_generator 함수를 사용합니다.
    # 예측 결과는 클래스별 확률 벡터로 출력되며,
    print("-- Predict --")
    print(test_generator.filenames)
    output = model.predict_generator(test_generator, steps=5)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(test_generator.class_indices)
    print(output)

    return model

load_model()