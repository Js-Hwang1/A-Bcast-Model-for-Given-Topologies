FROM ubuntu:22.04

LABEL author="jungshwang"
LABEL description="SimGrid SMPI broadcast simulation environment"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        python3 \
        python3-pip \
        python3-dev \
        libboost-all-dev \
        parallel \
    && pip3 install --no-cache-dir numpy \
    && rm -rf /var/lib/apt/lists/*

# Build SimGrid v3.35 from source (includes smpicc, smpirun)
RUN cd /opt \
    && git clone --depth 1 --branch v3.35 https://framagit.org/simgrid/simgrid.git \
    && cd simgrid \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
          -Denable_smpi=ON \
          -Denable_documentation=OFF \
          -Denable_compile_optimizations=ON \
          . \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / && rm -rf /opt/simgrid

ENV PATH=/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
