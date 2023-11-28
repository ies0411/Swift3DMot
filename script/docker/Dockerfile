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
# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake spconv-cu113 opencv-python open3d natsort

RUN echo "set enable-bracketed-paste off" >> ~/.inputrc   # Disable bracketed paste mode
RUN python3 -m pip install --user jupyter


RUN apt-get update && apt-get install -y openssh-client git
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN git clone https://github.com/subhadarship/kmeans_pytorch
RUN cd kmeans_pytorch && pip install --editable .

# sudo apt install python3.8
# sudo apt-get install python3.8 python3.8-dev python3.8-distutils python3.8-venv
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV DET=/opt/dec
ARG CACHEBUST=1

RUN --mount=type=ssh,id=repo.key,required git clone git@github.com:ies0411/3D_tracking.git /opt/dec
RUN --mount=type=ssh,id=repo.key,required python3 -m pip install --user -e /opt/dec


# pip install typing-extensions
RUN pip install kornia
RUN pip install --upgrade typing-extensions

# python -m pip install pip --upgrade
# apt install python3-pip
# pip install setuptools
# pip install Cython


# python3.8 -m pip install av2==0.2.0

# ENV AUTOLABEL_SRC=/opt/autolabel3d
# RUN --mount=type=ssh,id=repo.key,required git clone git@github.com:Superb-AI-Suite/autolabel3d.git $AUTOLABEL_SRC
# RUN --mount=type=ssh,id=repo.key,required git clone -b cal_3d git@github.com:Superb-AI-Suite/autolabel3d.git $AUTOLABEL_SRC
# RUN --mount=type=ssh,id=repo.key,required python3 -m pip install --user -e $AUTOLABEL_SRC