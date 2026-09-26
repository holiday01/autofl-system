# Pins the Python and library versions used in the paper's evaluation.
# Build:  docker build -t autofl:2.0 .
# Run:    docker run --rm -it autofl:2.0            (CPU; preflight validation is CPU-forced)
#         docker run --rm -it --gpus all autofl:2.0 (FL simulations used a GPU with AMP)
# GPU mixed-precision runs are not bitwise reproducible across hardware or drivers.
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/opt
RUN pip install torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install -r /tmp/requirements-lock.txt

# The code imports itself as the package `autofl`, so it must live in a directory of that name.
COPY . /opt/autofl
WORKDIR /opt/autofl
CMD ["bash"]
