import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt


def display_raw_or_predicted_mnist_data(images: np.ndarray, train_labels: np.ndarray, class_names: list, model: k.Model = None) -> None:
    random_sample = random.sample(range(10000), 12) if model else random.sample(range(60000), 12)
    for i in range(12):
        plt.subplot(3, 4, i+1)
        if model:
            img = images[random_sample[i]].reshape(1, 28, 28, 1)
            value = model.predict(img)
            predicted = np.argmax(value)
            print(f"Results provided by custom model:")
            nb = 0
            for j in range(len(class_names)):
                percentage = value[0][j]*100
                if percentage >= 0.01:
                    nb += 1
                    print(f"\t#{nb} {class_names[j]} @ {(percentage):.2f}%")
            plt.title(class_names[predicted])
        else:
            plt.title(class_names[train_labels[random_sample[i]]])
        plt.imshow(images[random_sample[i]], cmap=plt.cm.binary)
        plt.axis("off")
    # plt.savefig('../data/images/predicted_mnist_data_sample.png', bbox_inches='tight') if model else plt.savefig('../data/images/raw_mnist_data_sample.png', bbox_inches='tight')
    plt.show()


def build_custom_cnn_model(input_shape: tuple[int, int, int], num_classes: int) -> k.models.Sequential:
    model = k.models.Sequential([
        k.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Conv2D(64, (3, 3), activation="relu"),
        k.layers.MaxPooling2D((2, 2)),
        k.layers.Conv2D(64, (3, 3), activation="relu"),
        k.layers.Flatten(),
        k.layers.Dense(64, activation="relu"),
        k.layers.Dense(num_classes, activation="softmax")
    ])
    model.summary()
    return model


def display_model_accuracy(history: k.callbacks.History) -> None:
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    # plt.savefig('../data/images/model_accuracy.png', bbox_inches='tight')
    plt.show()


def use_custom_cnn_model_with_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = k.datasets.fashion_mnist.load_data() # Load the data from the fashion_mnist dataset
    train_images, test_images = train_images / 255.0, test_images / 255.0 # Normalize the pixel values to be between 0 and 1
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] # The names of the classes
    
    display_raw_or_predicted_mnist_data(train_images, train_labels, class_names) # Display a sample of the raw data
    
    model = build_custom_cnn_model((28, 28, 1), 10)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']) # Configure the model for training
    
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)) # Train the model
    display_model_accuracy(history)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) # Returns the accuracy and loss values of the model on the test_images
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
    
    display_raw_or_predicted_mnist_data(test_images, train_labels, class_names, model) # Display a sample of the predicted data


if __name__ == "__main__":
    use_custom_cnn_model_with_mnist_data()