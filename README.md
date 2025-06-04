# GNE-Vim
The paper: No-Reference Image Quality Assessment: Exploring Intrinsic Distortion Characteristics via Generative Noise Estimation with Mamba.
#
# Test Demo 
Please download model weights from [[Baidu](https://pan.baidu.com/s/1g5aAP4Ez3hx2-_3XK5A0yw), Password:yor0] and run
```
python test.py --dataset LIVE --name LIVE_test --ckpt  /path/of/your/model
```
#
# Requirement
* pip install torch torchvision pillow
* pip install -r vim/vim_requirements.txt
* pip install causal-conv1d==1.1.1
* cd Vision-Mamba/mamba-1p1p1  cp -rf mamba_ssm /xxx/anaconda3/lib/python3.xx/site-packages

