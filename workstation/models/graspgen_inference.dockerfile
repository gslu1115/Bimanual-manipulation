FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libosmesa6-dev build-essential ninja-build && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'numpy==1.26.4' 'setuptools==69.5.1' 'torchvision==0.16.0' 'h5py==3.11.0' hydra-core matplotlib meshcat 'scikit-learn==1.5.2' 'scipy==1.13.1' tensorboard 'trimesh==4.5.3' imageio opencv-python-headless 'pyrender==0.1.45' 'pyglet==2.1.6' 'PyOpenGL==3.1.0' 'diffusers==0.11.1' 'timm==1.0.15' 'huggingface-hub==0.25.2' addict 'yapf==0.40.1' tensorboardx 'torch-geometric==2.6.1' 'spconv-cu120==2.3.6' 'yourdfpy==0.0.56' webdataset pyzmq msgpack msgpack-numpy 'transformers==4.45.2'
RUN pip install --no-cache-dir torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
COPY pointnet2_ops /opt/pointnet2_ops
RUN pip install --no-build-isolation --no-deps /opt/pointnet2_ops
COPY . /opt/graspgen
RUN pip install --no-deps --no-build-isolation -e /opt/graspgen
WORKDIR /opt/graspgen
ENTRYPOINT ["python", "client-server/graspgen_server.py"]
