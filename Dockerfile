FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive

# Install packages for simpler
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion build-essential ca-certificates cmake curl git \
    htop unzip vim wget
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH


RUN wget -qO- https://astral.sh/uv/install.sh | sh
RUN conda install -c conda-forge libgl glib libvulkan-loader vulkan-tools vulkan-headers

# install dependencies
# RUN conda install -c conda-forge libgl glib libvulkan-loader vulkan-tools vulkan-headers
# RUN pip install uv

# docker build -t simpler_env .