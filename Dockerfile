FROM armv7/armhf-ubuntu:16.04

RUN apt-get update \
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


RUN cd / && git clone --recursive https://github.com/apache/incubator-mxnet.git --branch 1.6.0 mxnet
RUN mkdir /mxnet/build 

RUN cd /mxnet/build && /cmake-3.13.2/bin/cmake \
    -D VERBOSE=1 \
    -D USE_MKLDNN=0 \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D USE_OPENCV=1 \
    -D USE_CUDA=0 \
     ../
RUN cd /mxnet/build && make -j8
RUN ln -s /usr/local/lib/python3.5/dist-packages/mxnet /mxnet

RUN pip3 install boto3
RUN pip3 install numpy

RUN apt-get install -y gcc g++
RUN cd / && git clone https://github.com/opencv/opencv.git opencv
RUN cd /opencv && mkdir build
RUN cd /opencv/build &&  /cmake-3.13.2/bin/cmake \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/local/bin/python \
    -D VERBOSE=1 \
    -D CMAKE_BUILD_TYPE=RELEASE \    
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \ 
    -D OPENCV_SKIP_PYTHON_LOADER=OFF \
    ../ 
RUN cd /opencv/build && make -j8
RUN cd /opencv/build && make install

RUN cd /mxnet/python && python setup.py install
RUN cd /opencv/build/python_loader && python setup.py install

COPY code/image_inference.py /code/
COPY ml_model/* /ml_model/
RUN cd / && rm -R cmake*

ENTRYPOINT ["python", "/code/image_inference.py"]
