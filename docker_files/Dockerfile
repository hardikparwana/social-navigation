# Need this cuda image
FROM hardikparwana/cuda116desktop:ipopt
RUN apt-get update
RUN apt install -y python3-pip
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.11
RUN apt-get install -y python3.11-distutils
RUN apt-get install -y python3.11-dev
RUN apt install -y curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install numpy==1.22.3 matplotlib sympy argparse scipy==1.10.1
RUN python3.11 -m pip install cvxpy==1.2.0 cvxpylayers==0.1.5 gym==0.26.0 gym-notices==0.0.8 gym-recording==0.0.1 moviepy==1.0.3 jax==0.4.13 jaxlib==0.4.11 gpjax==0.5.9 optax==0.1.4 jaxopt 
RUN apt-get install pkg-config
RUN apt install -y wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
#RUN python3.11 -m pip install cyipopt==1.2.0
RUN python3.11 -m pip install jax==0.4.13 jaxlib==0.4.11
RUN python3.11 -m pip install diffrax==0.3.0
RUN python3.11 -m pip install pygame==2.3.0
RUN python3.11 -m pip install jax[cuda11_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python3.11 -m pip install --upgrade hj-reachability
RUN python3.11 -m pip install "cbf_opt>=0.6.0"
RUN python3.11 -m pip install "experiment-wrapper>=1.1"
