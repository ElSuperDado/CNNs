import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt

from random import randint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet


# choose a random image, displays it and returns its path
def choose_random_image() -> str:
    img_paths = []
    labels = []

    for dirpath, _, files in os.walk("../data/custom/predict"):
        for file in files:
            img_paths.append(f"{dirpath}/{file}")
            labels.append(dirpath.split("/")[-1])

    i = randint(0, len(img_paths)-1)
    img_path = img_paths[i]
    img = image.load_img(img_path)

    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")
    plt.show()

    return img_path


# model_name can be "vgg16", "xception" or "mobilenet"
def get_pretrained_model(model_name: str, image_size_targeted: tuple[int, int]) -> k.Model:
    match model_name:
        case "vgg16":
            model = VGG16
        case "xception":
            model = Xception
        case "mobilenet":
            model = MobileNet
        case _:
            print("Uncrecognised model")
            sys.exit(1)

    model = model(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=image_size_targeted,
        pooling=None,
        classes=1000,
        classifier_activation="softmax"
    )

    return model


def fine_tune_model(model: k.Model, nb_outputs: int, train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset, tl_epochs: int = 1, ft_epochs:int =1, ft_lr: float = 1e-5) -> None:
    # transfer learning model
    model.trainable = False

    x = model.output
    x = k.layers.GlobalAveragePooling2D()(x)
    x = k.layers.Dense(1024, activation="relu")(x)
    output = k.layers.Dense(nb_outputs, activation="softmax")(x)

    new_model = k.Model(inputs=model.input,outputs=output)

    new_model.compile(
        optimizer=k.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    new_model.fit(train_ds, epochs=tl_epochs)
    test_loss, test_accuracy = new_model.evaluate(valid_ds)
    print(f"{model_name} transfer learned:\n\tLoss: {test_loss} \n\tAccuracy: {test_accuracy}")

    # fine tuning model
    model.trainable = True
    new_model.compile(
        optimizer=k.optimizers.Adam(ft_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    new_model.fit(train_ds, epochs=ft_epochs)
    test_loss, test_accuracy = new_model.evaluate(valid_ds)
    print(f"{model_name} fine tuned:\n\tLoss: {test_loss} \n\tAccuracy: {test_accuracy}")
    return new_model


def import_dataset(category: str, image_size_targeted: tuple[int, int]) -> list[tf.data.Dataset]:
    match category:
        case "animals":
            train_ds, valid_ds = k.utils.image_dataset_from_directory(
                directory=f"../data/animals",
                labels="inferred",
                label_mode="categorical",
                validation_split=0.25,
                seed=1990,
                subset="both",
                image_size=image_size_targeted
            )

        case "fruits":
            train_ds = k.utils.image_dataset_from_directory(
                directory=f"../data/fruits/train",
                labels="inferred",
                label_mode="categorical",
                image_size=image_size_targeted
            )

            valid_ds = k.utils.image_dataset_from_directory(
                directory=f"../data/fruits/test",
                labels="inferred",
                label_mode="categorical",
                image_size=image_size_targeted
            )

        case "custom":
            train_ds, valid_ds = k.utils.image_dataset_from_directory(
                directory=f"../data/custom/train",
                labels="inferred",
                label_mode="categorical",
                validation_split=0.25,
                seed=1990,
                subset="both",
                image_size=image_size_targeted
            )

        case _:
            print("unrecognised image category")
            sys.exit(1)

    return train_ds, valid_ds


def predict(model: k.Model, image_path: str, class_names: list[str], image_size_targeted: tuple[int, int] = (224, 224, 3)) -> None:
    img = k.utils.load_img(
        image_path,
        target_size=image_size_targeted
    )

    img = k.utils.img_to_array(img)
    img = np.array([img])
    predictions = model.predict(img)
    print(predictions)

    print(f"Results provided by selected model:")
    for i in range(4):
        print(f"\t#{i+1} {class_names[i]} @ {(predictions[0][i]*100):.2f}%")


def get_class_names(category: str) -> list[str]:
    match category:
        case "animals":
            path = "../data/animal_names.txt"
        case "fruits":
            path = "../data/fruit_names.txt"
        case "custom":
            path = "../data/custom_names.txt"
        case _:
            print("unrecognised image category")
            sys.exit(1)

    class_names = []
    with open(path, "r") as f:
        for line in f.readlines():
            class_names.append(line.rstrip())

    return class_names


if __name__ == "__main__":    
    k.backend.clear_session()
    category = "custom"
    
    if len(sys.argv) == 5 :
        tl_epochs = int(sys.argv[2])
        ft_epochs = int(sys.argv[3])
        ft_lr = float(sys.argv[4])
    elif len(sys.argv) == 2:
        tl_epochs = 1
        ft_epochs = 1
        ft_lr = 1e-5
    else:
        print("Command error:\nUsage: pretrained_model.py [MODEL_NAME] [TRANSFER_LEARN_EPOCHS] [FINE_TUNING_EPOCHS] [FINE_TUNING_LEARNING_RATE]")
        print("[MODEL_NAME]                 -> vgg16 or xception or mobilenet")
        print("[TRANSFER_LEARN_EPOCHS]      -> optional")
        print("[FINE_TUNING_EPOCHS]         -> optional")
        print("[FINE_TUNING_LEARNING_RATE]  -> optional")
        sys.exit(1)
    
    model_name = sys.argv[1]
    if model_name == "xception":
        input_shape = (150, 150, 3)
    else:
        input_shape = (150, 150, 3)

    # importing data
    train_ds, valid_ds = import_dataset(category, input_shape[:2])
    class_names = get_class_names(category)
    
    # getting and training model
    base_model = get_pretrained_model(model_name, input_shape)
    new_model = fine_tune_model(base_model, 4, train_ds, valid_ds, tl_epochs, ft_epochs, ft_lr)
    
    # Testing results
    for i in range(5):
        img = choose_random_image()
        print(img)
        predict(new_model, img, class_names, input_shape[:2])
