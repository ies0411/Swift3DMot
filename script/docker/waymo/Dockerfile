# syntax=docker/dockerfile:1.2
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive
# Update nvidia GPG key: https://github.com/NVIDIA/nvidia-docker/issues/1631
COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
  rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
  dpkg -i cuda-keyring_1.0-1_all.deb && \
  rm cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y \
  python3-opencv ca-certificates python3-dev git wget sudo ninja-build  libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ libgl1-mesa-glx
RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"
RUN wget -O /root/get-pip.py https://bootstrap.pypa.io/get-pip.py && python3 /root/get-pip.py --user
RUN pip3 install --upgrade pip
# RUN pip3 install waymo-open-dataset-tf-2-5-0 --user
# SharedArray

RUN git clone https://github.com/subhadarship/kmeans_pytorch
RUN cd kmeans_pytorch && pip install --editable .

RUN pip install scipy scikit-image SharedArray opencv-python nuscenes-devkit==1.0.5 easydict spconv-cu113 numba open3d kornia==0.6.8

RUN pip3 install waymo-open-dataset-tf-2-6-0 --user
RUN pip install -U numpy
RUN pip install --upgrade typing-extensions





# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA