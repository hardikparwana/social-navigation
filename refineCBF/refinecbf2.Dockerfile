# Need this cuda image
FROM hardikparwana/cuda116desktop:ipopt
RUN apt-get update
RUN apt install -y python3-pip
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN python3 -m pip install jax[cuda11_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib==0.4.13

RUN python3 -m pip install hj-reachability
RUN python3 -m pip install "cbf_opt==0.6.0"
RUN python3 -m pip install "experiment-wrapper==1.1"
RUN python3 -m pip install notebook
RUN python3 -m pip install ipympl
