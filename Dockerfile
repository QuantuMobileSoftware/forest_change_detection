FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

RUN apt-get dist-upgrade 

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean


#Install GDAL and gdal*.py (global python==3.6)
RUN apt-get install python3-gdal python-gdal  -y


RUN apt install curl libpq-dev gdal-bin aptitude -y

RUN aptitude install libgdal-dev -y -o APT::Get::Fix-Missing=true

#Install python3.8 with miniconda
RUN curl -o /conda_install.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
RUN bash /conda_install.sh -b -p /miniconda
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init


# Needed for geopandas&shapely to work
RUN apt-get update && \
    apt-get install -y \
    git \
    libspatialindex-dev \
    ffmpeg \ 
    libsm6 \
    libxext6


RUN conda install jupyter==1.0.0
RUN pip install s2cloudless==1.7.0
RUN mkdir /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt


WORKDIR /code
COPY ./ .

CMD ["jupyter", "nbconvert", "--inplace", "--to=notebook", "--execute", "./Deforestation_inference.ipynb"]
