FROM ann-benchmarks

RUN apt update && apt install -y wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
RUN bash Anaconda3-2024.06-1-Linux-x86_64.sh -b

ENV PATH=/root/anaconda3/bin:$PATH

RUN python3 -m pip install ansicolors docker wheel requests
RUN conda install -y -c pytorch -c nvidia -c conda-forge faiss-gpu=1.8.0

# https://developpaper.com/a-pit-of-mkl-library-in-linux-anaconda/
ENV LD_PRELOAD=/root/anaconda3/lib/libmkl_core.so:/root/anaconda3/lib/libmkl_sequential.so 

RUN python3 -c 'import faiss; print(faiss.IndexFlatL2)'
