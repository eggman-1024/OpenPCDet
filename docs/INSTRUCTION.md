## 简介
`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection.

## 环境
- Linux (Ubuntu 22.04)
- Python 3.8 (后续创建conda环境时指定python=3.8)
- PyTorch 1.9.0 (后续通过pip安装)
- Spconv 2.x (后续通过pip安装)

## 安装
### 创建conda环境

```bash
conda create -n pcdet python=3.8
conda activate pcdet
```

### 安装PyTorch
去官网找链接[PyTorch 1.9.0+cu11.1](https://pytorch.org/get-started/previous-versions/#linux-and-windows-45)
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

> 由于可能需要复现一些较早的论文，为避免较新版本PyTorch与较早论文起冲突，因此未选择较新版本。

### 安装Spconv
去[Spconv 仓库](https://github.com/traveller59/spconv#prebuilt)安装
```bash
pip install spconv-cu114
```

### clone openpcdet仓库
> 我先fork后再clone fork后的仓库
```git
git clone https://github.com/eggman-1024/OpenPCDet.git
```

官方仓库：<u>https://github.com/open-mmlab/OpenPCDet.git</u>

### 根据requirements.txt安装
```bash
pip install -r requirements.txt
```

### 编译
```bash
python setup.py develop
```


## 训练模型
可以选择添加额外的命令行参数`--batch_size ${BATCH_SIZE}` 和 `--epochs ${EPOCHS}`

*You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters.*

- Train with multiple GPUs or multiple machines

```shell
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
# e.g.
sh scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/centerpoint.yaml
# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

- Train with a single GPU:

```shell
python train.py --cfg_file ${CONFIG_FILE}
# e.g.
python train.py --cfg_file cfgs/kitti_models/centerpoint.yaml
```


## Debug集合
### 训练时出现ModuleNotFoundError: No module named 'av2
原因：需要下载Argoverse2数据集

解决：可以不下载，而是将`OpenPCDet/pcdet/datasets/init. py`文件中的下面两行代码注释掉
- "from. argo2. argo2 dataset import Argo2Dataset" 
- "Argo2Dataset': Argo2Dataset "

参考:https://github.com/open-mmlab/OpenPCDet/issues/1328#issuecomment-1546529541
### 训练时会有/bin/sh: line 1: gpustat: command not found 的报错
原因：gpustat是python的一个包，缺少该包

解决：pip安装即可
(还未安装)
参考:https://blog.csdn.net/qq_29304033/article/details/123734119
