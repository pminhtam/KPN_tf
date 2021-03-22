## Attention Mechanism Enhanced Kernel Prediction Networks (AME-KPNs)
 Offical in [Attention-Mechanism-Enhanced-KPN](https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN)
 
 
The unofficial implementation of AME-KPNs in PyTorch, and paper is accepted by ICASSP 2020 (oral), it is available at [http://arxiv.org/abs/1910.08313](http://arxiv.org/abs/1910.08313).

## Data

Use [SIDD dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) to train. 

Have two folder : noisy image and ground true image

Input folder have struct :

```
/
    /noise
        /[scene_instance]
            /[image].PNG
    /gt
        /[scene_instance]
            /[image].PNG
```

## Train
This repo. supports training on multiple GPUs.  

Train



## Eval

Eval 


### News

## Name 


### Requirements
```
pip install -r requirments.txt
```

### Citation
```
https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN
```

```
@article{zhang2019attention,
    title={Attention Mechanism Enhanced Kernel Prediction Networks for Denoising of Burst Images},
    author={Bin Zhang and Shenyao Jin and Yili Xia and Yongming Huang and Zixiang Xiong},
    year={2019},
    journal={arXiv preprint arXiv:1910.08313}
}
```
