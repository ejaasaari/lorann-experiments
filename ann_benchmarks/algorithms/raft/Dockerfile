FROM ann-benchmarks

RUN apt update && apt install -y wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh -b

ENV PATH=/root/anaconda3/bin:$PATH

RUN python3 -m pip install ansicolors docker
RUN conda install -y -c rapidsai -c conda-forge -c nvidia pylibraft cuda-version=12.0 cupy
