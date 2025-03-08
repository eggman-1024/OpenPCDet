# A100服务器安装openpcdet

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