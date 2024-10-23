FROM ann-benchmarks

RUN apt update && apt install -y curl
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

RUN mkdir /home/app/micromamba
ENV MAMBA_ROOT_PREFIX=/home/app/micromamba
ENV MAMBA_EXE=/home/app/bin/micromamba
ENV PATH=/home/app/bin:$PATH

RUN micromamba shell init -s bash -r /home/app/micromamba
RUN micromamba create -y -n myenv -c rapidsai -c pytorch -c nvidia -c conda-forge pylibraft pytorch pytorch-cuda numpy h5py scikit-learn cupy cudnn cutensor nccl "libblas=*=*mkl"

ENV PATH=/home/app/micromamba/envs/myenv/bin:$PATH

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir ansicolors docker psutil wheel
RUN python -m pip install --no-cache-dir -U "jax[cuda12]"

RUN git clone https://github.com/ejaasaari/lorann

WORKDIR ./lorann
RUN git reset --hard e04b90e4bc05ffae75694ec84054ceb12d3d9198
WORKDIR ./python
RUN python3 setup.py install
WORKDIR /home/app
