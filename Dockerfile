# Start from the base image
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3


# Install the required packages
RUN pip3 install --no-cache-dir \
    matplotlib>=3.2.2 \
    numpy>=1.18.5 \
    opencv-python>=4.1.2 \
    Pillow>=7.1.2 \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.4.1 \
    torch>=1.7.0 \
    torchvision>=0.8.1 \
    tqdm>=4.41.0 \
    tensorboard>=2.4.1 \
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    onnx>=1.8.0 \
    onnx-simplifier>=0.3.6 \
    albumentations>=1.0.3 \
    thop

CMD ["/bin/bash"]
