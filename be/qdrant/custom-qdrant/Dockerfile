FROM qdrant/qdrant:gpu-nvidia-latest

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# Install software-properties-common so add-apt-repository works
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:kisak/kisak-mesa \
    && apt-get update \
    && apt-get upgrade -y \
	&& apt-get install -y \
    libxext6 \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    && apt-get install -y vulkan-tools mesa-vulkan-drivers
# RUN rm -rf /var/lib/apt/lists/*

COPY icd.d/ /etc/vulkan/icd.d/
COPY implicit_layer.d/ /etc/vulkan/implicit_layer.d/
COPY nvidia_icd.json /etc/vulkan/icd.d/
