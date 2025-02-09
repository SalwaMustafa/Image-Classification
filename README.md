# Image-Classification
This repository includes training a baseline AlexNet model and fine-tuning a pretrained InceptionV3 model on a dataset of 25k 150x150 images from 6 categories: buildings, forest, glacier, mountain, sea, and street. Additionally, a simple deployment using Streamlit is provided for model inference.


### Install python using Miniconda

1) Download and install Miniconda 
2) Create a new environment using the following command:
```bash
$ conda create -n Image-Classification python=3.9
```
3) Activate the environment:
```bash
$ conda activate Image-Classification
```

### Install the required packages

```bash
$ cd src
```
 
```bash
$ pip install -r requirements.txt
```

### Run the app 
```bash
$ streamlit run app.py
```