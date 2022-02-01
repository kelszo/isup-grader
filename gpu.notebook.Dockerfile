FROM nvcr.io/nvidia/pytorch:22.01-py3

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
    texlive-full \
    tini \
    wget 


# Libraries
RUN pip install --upgrade --no-cache-dir pip \
    && pip install --upgrade --no-cache-dir \
    albumentations \
    efficientnet-pytorch \
    humanize \
    imagecodecs \
    ImageHash \
    ipywidgets \
    joblib \
    opencv-python \
    openphi \
    openslide-python \
    pytorch-lightning \
    scikit-plot \
    seaborn \
    tifffile \
    tqdm \
    ujson \
    watermark

RUN pip install numpy --upgrade

COPY jupyter_notebook_config.json .
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]

ENV PYTHONPATH "${PYTHONPATH}:/lib/python3/"

CMD ["jupyter", "notebook", "--config=./jupyter_notebook_config.json"]