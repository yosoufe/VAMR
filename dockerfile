FROM nvidia/cudagl:11.4.1-devel-ubuntu20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y apt-utils && \
    apt-get install -y mesa-utils

RUN DEBIAN_FRONTEND=noninteractive \
        apt-get install -y \
        build-essential checkinstall cmake pkg-config yasm \
        gfortran git\
        libtiff5-dev \
        libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
        libxine2-dev libv4l-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        qt5-default libgtk2.0-dev libtbb-dev \
        libatlas-base-dev \
        libfaac-dev libmp3lame-dev libtheora-dev \
        libvorbis-dev libxvidcore-dev \
        libopencore-amrnb-dev libopencore-amrwb-dev \
        x264 v4l-utils \
        libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev libgflags-dev \
        libgphoto2-dev libhdf5-dev doxygen \
        python3.8 python3.8-dev

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
    cmake .. && \
    make -j`nproc` install

# install cudnn ibcudnn8_8.2.4.15-1+cuda11.4_amd64.deb
RUN apt-get install -y wget
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb & \
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb && \
    wait && \
    dpkg -i libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb &&\
    dpkg -i libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb


# opencv
RUN git clone -b 4.5.0 https://github.com/opencv/opencv_contrib.git & \
    git clone -b 4.5.0 https://github.com/opencv/opencv.git && \
    wait && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
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
      make -j`nproc` install

# install terminator
RUN apt-get update && apt-get autoremove -y \
    && apt-get install -y \
        python3-gi gir1.2-keybinder-3.0 gettext intltool dbus-x11 x11-apps\
        gobject-introspection \
        gir1.2-gtk-3.0 \
        libvte-2.91-dev \
        python-gobject \
        python3-gi-cairo \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
    && /usr/bin/python3 -m pip install psutil configobj && \
    git clone -b v1.92 --single-branch https://github.com/gnome-terminator/terminator.git \
    && cd terminator \
    && python3 setup.py build \
    && python3 setup.py install --record=install-files.txt \
    && cd ..

RUN apt-get install -y gdb cmake-curses-gui

RUN mkdir -p /code
WORKDIR /code

CMD [ "/usr/local/bin/terminator" ]