FROM ros:melodic-ros-core-bionic
ARG DEBIAN_FRONTEND=noninteractive
ARG RAVE_COMMIT=7c5f5e27eec2b2ef10aa63fbc519a998c276f908
ARG OSG_COMMIT=1f89e6eb1087add6cd9c743ab07a5bce53b2f480
ARG BOOST_SRC_DIR=~/git/boost_1_58_0
ENV QT_X11_NO_MITSHM=1

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    build-essential \
    git \
    gcc-4.8 \
    g++-4.8 \
    cmake \
    python-pip \
    python-scipy \
    coreutils \
    nano \
    vim \
    tmux \
    ipython \
    minizip \
    python-dev \
    python-h5py \
    python-numpy \
    python-scipy \
    python-sympy \
    qt4-dev-tools \
    wget \
    python-wheel \ 
    gedit

RUN apt-get install -q -y --no-install-recommends \
    libassimp-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libboost-date-time-dev \
    libbullet-dev \
    libfaac-dev \
    libglew-dev \
    libgsm1-dev \
    liblapack-dev \
    liblog4cxx-dev \
    libmpfr-dev \
    libode-dev \
    libogg-dev \
    libpcrecpp0v5 \
    libpcre3-dev \
    libqhull-dev \
    libqt4-dev \
    libsoqt-dev-common \
    libsoqt4-dev \
    libswscale-dev \
    libvorbis-dev \
    libx264-dev \
    libxml2-dev \
    libxvidcore-dev \
    libbz2-dev \
    libtinyxml-dev \
    liboctave-dev \
    libmpfi-dev \
    libfreetype6-dev \
    libflann-dev \
    libeigen3-dev \
    libann-dev \
    ann-tools \
    libccd-dev \
    octomap-tools \
    libminizip-dev \
    libcollada-dom2.4-dp-dev \
    libboost-python-dev \
    qt5-default minizip \
    liblapacke-dev \
    libnewmat10* \
    libgsl-dev \
    libcairo2-dev \
    # libjasper-dev \
    libpoppler-glib-dev \
    libsdl2-dev \
    libtiff5-dev \
    libxrandr-dev \
    libyaml-cpp-dev

RUN apt update && apt install -q -y --no-install-recommends \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool

RUN apt-get install -q -y --no-install-recommends \
    ros-melodic-chomp-motion-planner \
    ros-melodic-sbpl* \
    ros-melodic-ompl \
    ros-melodic-trac-ik-* \
    ros-melodic-srdfdom* \
    ros-melodic-qt-ros

RUN apt-get install -y --no-install-recommends \
    doxygen \
    octave \
    python-setuptools \
    mlocate \
    python-tk

RUN python2.7 -m pip install matplotlib==2.2.0 --user
RUN python2.7 -m pip install pydot==1.2.3
RUN python2 -m pip install enum34 --user
RUN python2 -m pip install networkx==2.2 --user

RUN pip install pyopengl tqdm

RUN mkdir -p ~/git; cd ~/git && \
   wget https://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download?use_mirror=autoselect -O ~/git/boost_1_58_0.tar.gz && \
   tar -xzf boost_1_58_0.tar.gz && \
   cd ~/git/boost_1_58_0/ && \
   ./bootstrap.sh --exec-prefix=/usr/local && \
   ./b2 -j `nproc` && \
   ./b2 -j `nproc` install threading=multi && \
   updatedb

#installing collada-dom
RUN mkdir -p ~/git; cd ~/git && \
    git clone https://github.com/rdiankov/collada-dom.git && \
    cd collada-dom && mkdir build && cd build && \
    cmake .. && \
    make -j `nproc` && \
    make install

# #install osg
RUN mkdir -p ~/git; cd ~/git && \
    git clone https://github.com/openscenegraph/OpenSceneGraph.git && \
    cd OpenSceneGraph; git reset --hard 1f89e6eb1087add6cd9c743ab07a5bce53b2f480 && \
    mkdir build; cd build && \
    cmake -DDESIRED_QT_VERSION=4 .. && \
    make -j `nproc` && make install && make install_ld_conf

# #install FCL
RUN mkdir -p ~/git; cd ~/git && \
    git clone https://github.com/flexible-collision-library/fcl && \
    cd fcl; git reset --hard 0.5.0 && \
    mkdir build; cd build && \
    cmake .. && \
    make -j `nproc` && \
    make install

# #installing Openrave
RUN pip install --upgrade --user sympy==0.7.1 && pip install numpy
RUN mkdir -p ~/git; cd ~/git && \
	git clone -b production https://github.com/rdiankov/openrave.git && \
    cd openrave; git reset --hard 7c5f5e27eec2b2ef10aa63fbc519a998c276f908 && \
    mkdir build; cd build && \
    cmake -DODE_USE_MULTITHREAD=ON -DCMAKE_CXX_STANDARD=11            \
        -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_ROOT=/usr/local/ .. && \
    make -j `nproc` && \
    make install

# #setting up workspace
RUN apt-get update && apt-get install -y libompl12 python-catkin-tools && \
    apt update && apt install -y libompl-dev && \
    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc && \
    echo 'alias sb="source ~/.bashrc"' >> ~/.bashrc

RUN cd ~/git && \
    git clone https://github.com/AAIR-Lab/or_catkin.git && \
    mkdir -p ~/tmp_catkin_ws/src/ && \
    rm -rf ~/git/or_catkin/openrave-installation/ && \
    mv or_catkin/* ~/tmp_catkin_ws/src/ && \
    rm -rf or_catkin/
 
RUN source /opt/ros/melodic/setup.bash && \
    cd ~/tmp_catkin_ws/ && \
    catkin_make && \
    echo "source ~/tmp_catkin_ws/devel/setup.bash" >> ~/.bashrc

#install x11 files
RUN apt-get update
RUN apt-get install -y xauth
RUN apt-get install -y xorg openbox

RUN apt install ros-melodic-gazebo-msgs
RUN pip install --upgrade pip setuptools wheel
RUN pip install opencv-python==4.2.0.32
