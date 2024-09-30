FROM nvcr.io/nvidia/jax:23.10-py3
RUN apt-get update
RUN apt-get install -y gedit
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN pip3 install PyQt5
RUN pip3 install matplotlib

## NOTE: gpjax version dependent on JAX version. Should change depending on which jax nvidia image has been pulled up
# with nvcr.io/nvidia/jax:23.10-py3
# RUN pip3 install gpjax==0.8.0 


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libqt5gui5
RUN apt-get install -y texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng cm-super
RUN apt install -y vim

# RUN pip3 install setuptools==58.0.4
RUN pip3 install empy==3.3.4

RUN python3 -m pip install --upgrade setuptools
RUN pip3 install polytope