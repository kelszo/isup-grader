FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

RUN apt-get update && apt-get -y update

RUN apt-get install -y \
    build-essential \ 
    ffmpeg \ 
    gdebi \
    libgl1-mesa-glx \
    libjpeg-turbo8 \
    libsm6 \ 
    libxext6 \
    openslide-tools \ 
    python3-dev \
    python3-pip \ 
    python3.8 \ 
    tini \
    wget 

RUN pip3 install pip --upgrade --quiet

# Notebook
RUN pip3 install --upgrade \
    jupyter \
    ipywidgets \
    watermark

# Pytorch
RUN pip3 install --upgrade \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Libraries
RUN pip install --upgrade --no-cache-dir pip \
    && pip install --upgrade --no-cache-dir \
    albumentations \
    efficientnet-pytorch \
    humanize \
    imagecodecs \
    joblib \
    keras \
    numpy \
    opencv-python \
    openphi \
    openslide-python \
    pandas \
    Pillow \
    pytorch-lightning \
    scikit-image \
    scikit-learn \
    scikit-plot \
    seaborn \
    tensorflow \
    tifffile \
    tifffile \
    torch \
    tqdm \
    xgboost 

COPY jupyter_notebook_config.json .
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]

ENV PYTHONPATH "${PYTHONPATH}:/lib/python3/"

CMD ["jupyter", "notebook", "--config=./jupyter_notebook_config.json"]