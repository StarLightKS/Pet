"""
Скрипт для создания модели распознавания эмоций питомцев по фото.
Использует архитектуру MobileNetV2 + Dense слои (как в Pet Emotion Recognition.ipynb).

Для точных предсказаний запустите обучение в ноутбуке Pet Emotion Recognition.ipynb
с датасетом https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Скрыть предупреждения TensorFlow

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def create_pet_emotion_model(output_path='pet_emotion.h5'):
    """Создаёт модель с архитектурой из ноутбука и сохраняет в .h5"""
    print("Загрузка MobileNetV2 (предобучен на ImageNet)...")
    conv_base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    conv_base.trainable = False

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Сохранение модели в {output_path}...")
    model.save(output_path)
    print(f"Модель сохранена: {output_path}")
    print("\nПримечание: Классификатор не обучен. Для точных предсказаний обучите модель")
    print("в ноутбуке Pet Emotion Recognition.ipynb на датасете pet emotion.")
    return model

if __name__ == '__main__':
    create_pet_emotion_model()
