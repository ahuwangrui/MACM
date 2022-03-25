![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![pytorch](https://img.shields.io/badge/pytorch-1.2-blue.svg)
![Author](https://img.shields.io/badge/wangrui-AHU-orange.svg)

##Prepare
## Prerequisites
- Python 3.6
- GPU Memory >= 11G
- Numpy
- pytorch 1.2
- OpenCV  4.5

## Step 1: Download Datasets and Camstyle aug-data.
- Market-1501 [[BaiduYun]](http://pan.baidu.com/s/1ntIi2Op) [[GoogleDriver]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing) CamStyle (generated by CycleGAN) [[GoogleDriver]](https://drive.google.com/open?id=1klY3nBS2sD4pxcyUbSlhtfTk9ButMNW1) [[BaiduYun]](https://pan.baidu.com/s/1NHv1UfI9bKo1XrDx8g70ow) (password: 6bu4)
   
- DukeMTMC-reID [[BaiduYun]](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) (password: bhbh) [[GoogleDriver]](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) CamStyle (generated by CycleGAN) [[GoogleDriver]](https://drive.google.com/open?id=1tNc-7C3mpSFa_xOti2PmUVXTEiqmJlUI) [[BaiduYun]](https://pan.baidu.com/s/1NHv1UfI9bKo1XrDx8g70ow) (password: 6bu4)
 
  *Camstyle images generated by GAN are provided by [Zhun Zhong](https://github.com/zhunzhong07)*

## Step 2: Unzip datasets and create folder.
unzip Market-1501 and DukeMTMC-reID datasets to data dir following the below structure.
 
```
├── MACM
   ├── data
      ├── market
         ├── bounding_box_train
         ├── bounding_box_test
         ├── query
      ├── duke
         ├── bounding_box_train
         ├── bounding_box_test
         ├── query
``` 


## Step 3: Generate VotingMask data by SCHP.
1.Down load [SCHP](https://github.com/PeikeLi/Self-Correction-Human-Parsing) and build the required environment following SCHP'`readme.md`.

2.Down pretrained model [LIP](https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing) and put it into `SCHP/models/lip`

3.Replace SCHP'`simple_extractor.py` by our `simple_extractor.py`.

4.Copy data in `MCAM/data/market/bounding_box_train` to `/SCHP/input` and run ```python simple_extractor.py```.

5.Copy VoingMask data in `SCHP/output` to `MACM/data/market/bouding_box_train`.

6.Do the same thing to ***duke***.

## Step 4: Add Camstyle aug-data and construct dataset for pretrain and cluster.
1.unzip Camstyle aug-data of Market-1501 and DukeMTMC-reID and copy them to `MACM/data/market/bounding_box_train` and `MACM/data/duke/bounding_box_train` .

2.run ``` python prepare.py``` for ***market***.

3.modify download_path as *duke* in prepare.py and run ``` python prepare.py``` for ***duke***.

Then,if everything goes well,you will get directory like below structure.
```
├── MACM
   ├── data
      ├── market
         ├── bounding_box_train
         ├── bounding_box_test
         ├── query
         ├── pytorch
             ├── gallery
             ├── query
             ├── train_aug
             ├── train_aug_newID
             ├── train_ori
             ├── train_ori_newID
``` 

##Training and Evaluate
For market to duke:
```
python train_m2d.py
```
For duke to market:
```
python train_d2m.py
```

