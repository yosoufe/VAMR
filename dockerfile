# FROM nvidia/cudagl:11.4.1-devel-ubuntu20.04
FROM cudagl:11.7.0-devel-ubuntu20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install -y apt-utils \
            mesa-utils \
            curl \
            wget \
            build-essential \
            checkinstall \
            cmake \
            pkg-config \
            yasm \
            gfortran git\
            libtiff5-dev \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev \
            libdc1394-22-dev \
            libxine2-dev \
            libv4l-dev \
            libgstreamer1.0-dev \
            libgstreamer-plugins-base1.0-dev \
            qt5-default \
            libgtk2.0-dev \
            libtbb-dev \
            libatlas-base-dev \
            libfaac-dev \
            libmp3lame-dev \
            libtheora-dev \
            libvorbis-dev \
            libxvidcore-dev \
            libopencore-amrnb-dev \
            libopencore-amrwb-dev \
            x264 \
            v4l-utils \
            libprotobuf-dev \
            protobuf-compiler \
            libgoogle-glog-dev \
            libgflags-dev \
            libgphoto2-dev \
            libhdf5-dev \
            doxygen \
            python3.8 \
            python3.8-dev \
            gdb \
            cmake-curses-gui

RUN ln -s /usr/bin/python3.8 /usr/bin/python

# install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python

RUN python3 -m pip install numpy

RUN mkdir 3rdparty

WORKDIR 3rdparty

# install eigen
RUN git clone -b 3.4 https://gitlab.com/libeigen/eigen.git && \
    cd eigen && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Debug .. && \
    make -j`nproc` install && \
    cd ../.. && \
    rm -rf eigen

# install cudnn
# RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.4.1.50-1+cuda11.6_amd64.deb \
#     & wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.4.1.50-1+cuda11.6_amd64.deb \
#     && wait \
#     && dpkg -i libcudnn8_8.4.1.50-1+cuda11.6_amd64.deb \
#     && dpkg -i libcudnn8-dev_8.4.1.50-1+cuda11.6_amd64.deb \
#     && rm -rf libcudnn*.deb

COPY docker_extra/cudnn-* .
RUN dpkg -i cudnn-local-repo-* \
    && cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get install libcudnn8 \
        libcudnn8-dev \
        libcudnn8-samples

# vtk, required for the 3d viz in opencv
RUN git clone https://github.com/Kitware/VTK.git -b v9.0.3 && \
    cd VTK && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON .. && \
    make -j`nproc` install && \
    cd ../.. && \
    rm -rf VTK


# opencv
RUN git clone -b 4.5.5 https://github.com/opencv/opencv_contrib.git & \
    git clone -b 4.5.5 https://github.com/opencv/opencv.git && \
    wait && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Debug \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_VTK=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D CUDA_ARCH_BIN=7.5,8.6 \
      -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
      -D CUDNN_INCLUDE_DIR=/usr/include/ \
      -D PYTHON3_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") \
      .. && \
    make -j`nproc` install && \
    cd ../.. && \
    rm -rf opencv opencv_contrib

# install google tests
RUN git clone https://github.com/google/googletest.git && \
    cd googletest && cmake . && \
    make -j`nproc` install && \
    cd .. && rm -rf googletest

# terminator
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y terminator

# # install nsight
# RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-nsight-11-7_11.7.50-1_amd64.deb \
#     & wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-nsight-compute-11-7_11.7.0-1_amd64.deb \
#     & wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/nsight-compute-2022.2.0_2022.2.0.13-1_amd64.deb \
#     & wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/nsight-systems-2022.1.3_2022.1.3.3-1_amd64.deb \
#     & wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-nsight-systems-11-7_11.7.0-1_amd64.deb \
#     && wait \
#     && apt-get install -y default-jre \
#     && dpkg -i cuda-nsight-11-7_11.7.50-1_amd64.deb \
#     && dpkg -i nsight-compute-2022.2.0_2022.2.0.13-1_amd64.deb \
#     && dpkg -i cuda-nsight-compute-11-7_11.7.0-1_amd64.deb \
#     && dpkg -i nsight-systems-2022.1.3_2022.1.3.3-1_amd64.deb \
#     && dpkg -i cuda-nsight-systems-11-7_11.7.0-1_amd64.deb \
#     && rm -rf *nsight*.deb

# install nsight
COPY docker_extra/nsight-systems-* .
RUN dpkg -i nsight-systems-*

RUN mkdir -p /code
WORKDIR /code/exercises

CMD [ "/usr/bin/terminator" ]
# CMD [ "bash" ]
