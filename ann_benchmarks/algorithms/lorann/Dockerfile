FROM ann-benchmarks

RUN apt update && apt install -y libtcmalloc-minimal4 

RUN git clone https://github.com/ejaasaari/lorann

WORKDIR ./lorann
RUN git reset --hard e04b90e4bc05ffae75694ec84054ceb12d3d9198
WORKDIR ./python
RUN python3 setup.py install
WORKDIR /home/app

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

ENV OMP_NUM_THREADS=1
