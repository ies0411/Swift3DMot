# syntax=docker/dockerfile:1.2
FROM python:3.8
ENV DEBIAN_FRONTEND noninteractive
# RUN python -m pip install --upgrade pip
RUN pip install av2==0.2.0
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


RUN git clone https://github.com/subhadarship/kmeans_pytorch
RUN cd kmeans_pytorch && pip install --editable .

RUN pip install -U numpy
RUN pip install SharedArray tqdm  scikit-learn kornia scikit-image  easydict spconv-cu113 numba pyyaml tensorboardX
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


#for lyft
RUN pip install -U lyft_dataset_sdk==0.0.8
RUN pip install open3d


ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV FORCE_CUDA="1"

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN  /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN   rm ~/miniconda.sh
# RUN  /opt/conda/bin/conda clean -tipsy
RUN  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# echo "conda activate my_env" >> ~/.bashrc
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
RUN pip install kornia==0.6.8
