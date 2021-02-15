FROM arm32v7/ubuntu:latest

RUN apt-get update \
  && apt-get install -y software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt update \
  && apt install -y python3.7 \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install -y git

RUN apt remove cmake

ADD https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz cmake-3.13.2.tar.gz
RUN tar xf cmake-3.13.2.tar.gz
RUN cd cmake-3.13.2 && ./configure
RUN cd cmake-3.13.2 && make -j4

RUN apt-get install -y libopenblas-dev libopencv-dev
RUN apt-get install -y build-essential ccache ninja-build

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install mxnet
RUN apt-get install python3-numpy
RUN apt-get install python3-opencv

COPY code/image_inference.py /code/
COPY ml_model/* /ml_model/
RUN cd / && rm -R cmake*

ENTRYPOINT ["python3", "/code/image_inference.py"]
