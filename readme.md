
# CityLayout

👆 CityLayout is a generative city 2d layout workflow, you can custom your OSM (openstreetmap) dataset by using
provided scripts automatically. The dumped raw data will be stored as GeoJSON format and after preprocessing, it will be
saved as one-hot encoded numpy array (eg. 0: road, 1: building, 2: water, 3: park, 4: etc). You can training with your 
custom dataset or just use pretrained model to generate city layout. 

## Todo

- [x] Release code
- [ ] Release dataset preprocess script
- [ ] Release pretrain ckpt
- [ ] Release dataset
- [ ] Webui demo

## Setup environment

```bash
# conda env
conda create -f environment.yml
# pip install
pip install -r requirements.txt
```

## Download pretrain ckpt



## Download dataset (optional)

We provide our used dataset for training and evaluation, you can download it from ####
and unzip it to `data` folder.

```bash
# download dataset
mkdir data && cd data
wget ###### 
unzip ######
```

The data dir may look like this:

```bash
data
├── train
│   ├── cityname-0-0.h5
│   ├── cityname-0-1.h5
│   ├── cityname-0-2.h5
...
├── val
│   ├── cityname-0-0.h5
...
├── test
│   ├── cityname-0-0.h5
...
``` 
The provided data has been preprocessed and saved as h5 format, you can directly use it for training and evaluation.

## Download dataset and preprocess

```bash
# download osm data

```

## Train model

```bash

```

## Generate city layout

```bash

```

## Acknowledgement

This project is inspired by [CityGen](https://github.com/rese1f/CityGen) we partially reproduce their work and make some
improvements. If you find this project helpful, please consider to cite their work.

```
@article{deng2023citygen,
  title={CityGen: Infinite and Controllable 3D City Layout Generation},
  author={Deng, Jie and Chai, Wenhao and Guo, Jianshu and Huang, Qixuan and Hu, Wenhao and Hwang, Jenq-Neng and Wang, Gaoang},
  journal={arXiv preprint arXiv:2312.01508},
  year={2023}
}
```


