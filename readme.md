
# CityLayout

👆 CityLayout is a generative city 2d layout workflow, you can custom your OSM (openstreetmap) dataset by using
provided scripts automatically. The dumped raw data will be stored as GeoJSON format and after preprocessing, it will be
saved as one-hot encoded numpy array (eg. 0: road, 1: building, 2: water, 3: park, 4: etc). You can training with your 
custom dataset or just use pretrained model to generate city layout. 

## Todo

- [x] Release code
- [x] Release dataset preprocess script
- [x] Release pretrain ckpt
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

You can download our pretrained model from [here](https://1drv.ms/f/s!Ap2hsgjizYNEj9saA4W4OC-1ma6saA?e=frKGC6)
You can find three different ckpt folders, the details are as follows:

| ckpt | Description |
| --- | --- |
| CityGen/best_ckpt.pth or latest_ckpt.pth | The pretrained model under [CityGen](https://github.com/rese1f/CityGen)'s setting |
| normal/best_ckpt.pth or latest_ckpt.pth | The pretrained model for citylayout random generation |
| Completion/best_ckpt.pth or latest_ckpt.pth | The pretrained model for citylayout completion |

All the ckpts mentioned above are trained with the same dataset setting , the training/sampling citylayout include building, road, and waterbody.

After downloading the ckpt, create soft link to `ckpts` folder.

```bash
ln -s /path/to/ckpt ckpts
```


## Download dataset (optional & not released yet)

We provide our used dataset for training and evaluation, you can download it from ####
and create soft link to `data` folder.

```bash
ln -s /path/to/data data
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

**This tutorial will help you create yout own osm training data from scratch**

### First: get target position (lat, lon) by runing script `get_locations`

the query list is stored in config/data/city_landmark.yaml
```bash
# get target position 
python scripts/osm/get_locations.py
```

### Second: download osm data by runing script `get_city_osm.py`

the default config for downloading osm data is stored in config/data/osm_cities.yaml,
including some path like input (city coordinates), output (osm data), and some other parameters like citylayout patch size(pixel), etc.


```bash
# download osm data
python scripts/osm/get_city_osm.py

```

## Train model

```bash

```

## Generate city layout
You can generate citylayout from gaussian noise by using normal ckpt, or you can generate citylayout from existing citylayout by using citygen ckpt with inpainting or outpainting mode, lastly, you can refine the citylayout by using completion ckpt.

### Generate city layout from gaussian noise
```bash
# random sample
python src/train_lightning.py -cn=refactoring_sample

# with refiner 

python src/train_lightning.py -cn=refactoring_sample CityDM.Test.use_completion=True

```

### Generate city layout from existing city layout
```bash

# inpainting
python src/train_lightning.py -cn=refactoring_sample_citygen CityDM.Test.sample_type=Inpainting

# outpainting

python src/train_lightning.py -cn=refactoring_sample_citygen CityDM.Test.sample_type=Outpainting

```


for compuational cluster, you can use the following command to generate city layout
```bash
# random sample
sbatch scripts/run/sample_normal.sh 
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


