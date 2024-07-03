# Simple CNNs implementation

**Authors**: Ludovico Grabau & Damian Boquete Costa

**Last modification**: 14.12.2022

### Description:
This work implements various Convolutional Neural Networks (CNN) with Python and Keras. 

It is divided into two parts:
- A custom implementation of a CNN, trained on FashionMNIST data.
- Existing pre-trained models that have been fine-tuned to work on a custom image dataset.

[`rapport.pdf`](rapport.pdf) is the paper (only in French) describing this work in depth.

### How to use the program:

#### Launching the pretrained models program

If it's the first time running it on the machine:
```sh
python3 -m venv .venv                 # create a virtual environment
source .venv/bin/activate             # activate the virtual environment
pip install -r requirements.txt       # install project dependencies
cd data/
chmod +x setup.sh                     # add necessary permissions to the installation program
./setup.sh                            # run the program that installs data
cd ../src/
python3 pretrained_model.py [OPTIONS] # launch the program
```

Otherwise, it can be launched as follows:
```sh
cd src/
python3 pretrained_model.py [OPTIONS] # launch the program
```

The available options for the `python3 pretrained_model.py` command are:
```sh
# options order
pretrained_model.py [MODEL_NAME] [TRANSFER_LEARN_EPOCHS] [FINE_TUNING_EPOCHS] [FINE_TUNING_LEARNING_RATE]

# options
[MODEL_NAME]                 -> (Required)  String Options: vgg16 or xception or mobilenet (case-sensitive)
[TRANSFER_LEARN_EPOCHS]      -> (Optional)  Int    Positive integer (e.g., 5)
[FINE_TUNING_EPOCHS]         -> (Optional)  Int    Positive integer (e.g., 5)
[FINE_TUNING_LEARNING_RATE]  -> (Optional)  Float  Small positive floating-point number (scientific notation) (e.g., 1e-5)
```

#### Launching the custom CNN model program

If it's the first time running it on the machine:
```sh
python3 -m venv .venv                 # create a virtual environment
pip install -r requirements.txt       # install project dependencies
source .venv/bin/activate             # activate the virtual environment
cd src/                               # navigate to the src folder
python3 custom_model.py               # launch the program
```

Otherwise, it can be launched as follows:
```sh
cd src/                               # navigate to the src folder
python3 custom_model.py               # launch the program
```
