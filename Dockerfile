FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive

# Install packages for simpler
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     bash-completion build-essential ca-certificates cmake curl git \
#     htop unzip vim wget \
#     libvulkan1 libvulkan-dev vulkan-tools xvfb \
#     libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglib2.0-0
# RUN rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion build-essential ca-certificates cmake curl git \
    htop libegl1 libxext6 libjpeg-dev libpng-dev  libvulkan1 rsync \
    tmux unzip vim vulkan-utils wget xvfb pkg-config ffmpeg \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglib2.0-0
RUN rm -rf /var/lib/apt/lists/*

# libvulkan1 libvulkan-dev vulkan-tools xvfb ffmpeg これだけでもいいかも？

RUN wget -qO- https://astral.sh/uv/install.sh | sh

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY nvidia/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# install dependencies
# RUN conda install -c conda-forge libgl glib libvulkan-loader vulkan-tools vulkan-headers
# RUN pip install uv

# docker build -t simpler_env .