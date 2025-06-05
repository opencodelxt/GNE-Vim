# GNE-Vim
The paper: No-Reference Image Quality Assessment: Exploring Intrinsic Distortion Characteristics via Generative Noise Estimation with Mamba.
#
# Test Demo 
Please download model weights from [[Baidu](https://pan.baidu.com/s/1FwtFNnMGRb3ZR_IOxvvzYQ), Password:fwuh] and run
```
python predict.py --dataset TID2013 --name TID2013_test --ckpt  /path/of/your/model
```
#
# Requirement
* pip install torch torchvision pillow
* pip install -r vim/vim_requirements.txt
* pip install causal-conv1d==1.1.1
* cd Vision-Mamba/mamba-1p1p1  cp -rf mamba_ssm /xxx/anaconda3/lib/python3.xx/site-packages

