# Need this cuda image
FROM hardikparwana/cuda118desktop:ros-humble-rmf

RUN apt-get update
RUN apt-get install -y wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.11
RUN apt-get install -y python3.11-distutils
RUN apt-get install -y python3.11-dev
RUN apt-get install -y curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install --upgrade setuptools
RUN python3.11 -m pip install numpy matplotlib
RUN python3.11 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib
RUN python3.11 -m pip install matplotlib==3.7.1 pillow==9.5.0 kiwisolver==1.4.4 polytope

WORKDIR /home/social-navigation
