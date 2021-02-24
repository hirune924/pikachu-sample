#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="@hirune924"

RUN conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch

RUN apt-get update --fix-missing && apt-get install -y \
    git libopencv-dev python-openslide unzip && \
    apt-get install -y libopenmpi-dev
    #&& \
    #apt-get clean && \
    #apt-get autoremove && \
    #rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
    
RUN pip install --upgrade pip && \
    pip install pandas scikit-image opencv-python openslide-python \
                scikit-learn pytorch-lightning imagecodecs tifffile \
                pretrainedmodels catalyst[all] segmentation-models-pytorch timm hydra-core \
                scikit-plot && \
    pip install -U neptune-client && \
    pip install -U git+https://github.com/albumentations-team/albumentations && \
    pip install kornia captum && \
    pip install fairscale && \
    pip install deepspeed mpi4py
