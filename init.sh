##================ 训练安装环境=================##  ncvv 11.3尤其重要
conda create -n galic python=3.7
conda activate galic
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmdet==2.25.3
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install einops==0.6.0
pip install fairscale==0.4.5
pip install kornia==0.6.0
pip install tensorboardX
conda install cudatoolkit=11.3
pip install cupy-cuda113
pip install spikingjelly
pip install natsort
pip install fvcore
pip install timm