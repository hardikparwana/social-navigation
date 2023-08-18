FROM hardikparwana/cuda116desktop:ipopt-ros1
COPY requirements.txt /home/requirements.txt
WORKDIR /home/
RUN pip3 install -r requirements.txt

RUN apt install -y python3.9 python3.9-dev python3.9-distutils
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -r requirements.txt
