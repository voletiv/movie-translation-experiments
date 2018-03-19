# movie-translation-experiments

My experiments with translation of lip movement in movies from one dialogue to another, possibly across languages.

Tentative pipeline:

![alt text](Movie_Translation.png "IMAGE NOT FOUND")


# INSTALLATION

The requirements have been listed in requirements.txt file.

Apart from them, to get 3D landmarks, the following repository is required: [face_alignment](https://github.com/1adrianb/face-alignment) - 2D and 3D Face alignment library build using pytorch.

The installation instructions for this for Ubuntu are (according to the [face_alignment](https://github.com/1adrianb/face-alignment) page):

- ```pip3 install numpy pyyaml mkl setuptools cmake gcc cffi```
    - I had problems installing gcc, but I skipped it because it was already installed (and might not be possible via pip)

(In case conda is not installed: [miniconda3](https://conda.io/miniconda.html))

- ```conda install -c soumith magma-cuda80``` (Depending on CUDA version: can be checked via ```nvcc -V```; I installed ```conda install -c soumith magma-cuda91``` for CUDA 9.1)

- Install Pytorch
    - git clone --recursive https://github.com/pytorch/pytorch
    - python3 setup.py install

- Install face_alignment
    - git clone https://github.com/1adrianb/face-alignment
    - sudo pip3 install -r requirements.txt
    - python3 setup.py install




