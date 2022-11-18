# 安装环境
## 创建虚拟环境
conda create -n vitea python=3.7 -y
conda activate obbdetection

## 安装Pytorch和Torchvision(cuda11.0)
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

## 安装ViTEA
- 安装 BboxToolkit
cd OBBDetection
cd BboxToolkit
pip install -v -e .  # or "python setup.py develop"
cd ..
- 安装mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# 编译安装ViTEA
pip install -r requirements/build.txt
pip install mmpycocotools
pip install -v -e .  # or "python setup.py develop"

# 训练
python -m torch.distributed.launch --nproc_per_node=1 --master_port=50002 tools/train.py \
configs/amy_configs/hrsc2016/faster_rcnn_orpn_our_imp_vitae_fpn_3x_hrsc.py \
--launcher 'pytorch' --options 'find_unused_parameters'=True



python tools/train.py configs/amy_configs/hrsc2016/faster_rcnn_orpn_our_imp_vitae_fpn_3x_hrsc.py 
python tools/train.py configs/amy_configs/hrsc2016/s2anet_our_rsp_vitae_fpn_1x_hrsc.py
python tools/train.py configs/amy_configs/hrsc2016/s2anet_r50_fpn_1x_hrsc.py

