# Need this cuda image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get update & apt-get install -y tmux gedit vim
RUN apt install -y python3-pip
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN python3 -m pip install jax[cuda12] 

RUN python3 -m pip install hj-reachability
RUN python3 -m pip install "cbf_opt>=0.6.0"
RUN python3 -m pip install "experiment-wrapper>=1.1"
RUN python3 -m pip install notebook
RUN python3 -m pip install ipympl

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.11
RUN apt-get install -y python3.11-distutils
RUN apt-get install -y python3.11-dev
RUN apt-get install -y curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install jax[cuda12]
RUN python3.11 -m pip install hj-reachability
RUN python3.11 -m pip install "cbf_opt>=0.6.0"
RUN python3.11 -m pip install "experiment-wrapper>=1.1"
RUN python3.11 -m pip install notebook
RUN python3.11 -m pip install ipympl

