# Miformer: A Minus Inverted Transformer Fed by Historical-Future Interactions for Trajectory Prediction
This repository is the official implementation of Miformer: A Minus Inverted Transformer Fed by Historical-Future Interactions for Trajectory Prediction.

## Table of Contents
+ [Setup](#setup)
+ [Datasets](#datasets)
+ [Training](#training)
+ [Validation](#validation)
+ [Testing](#testing)
+ [Acknowledgements](#acknowledgements)

## Setup
Clone the repository and set up the environment:
```
git clone https://github.com/Morphlingxxx/Miformer-Trajectory-Prediction.git
cd Miformer-Trajectory-PredictionP
conda create -n Miformer python=3.10
conda activate Miformer
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
conda install pytorch-lightning
pip install itransformer
```
*Note:* For compatibility, you may experiment with different versions, e.g., PyTorch 2.3.1 has been confirmed to work.

*Importatnt*
Replace itransformer.py in the conda environment with iTransformerMinus.py

## Datasets

<details>
<summary><b>Argoverse</b></summary>
<p>

1. Download the [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html#download-link). After downloading and extracting the tar.gz files, organize the dataset directory as follows:

```
/path/to/Argoverse_root/
├── train/
│   └── data/
│       ├── 1.csv
│       ├── 2.csv
│       ├── ...
└── val/
    └── data/
        ├── 1.csv
        ├── 2.csv
        ├── ...
```

2. Install the [Argoverse API](https://github.com/argoverse/argoverse-api).

</p>
</details>

<details>
<summary><b>INTERACTION</b></summary>
<p>

1. Download the [INTERACTION Dataset v1.2](https://interaction-dataset.com/). Here, we only need the data for the multi-agent tracks. After downloading and extracting the zip files, organize the dataset directory as follows:

```
/path/to/INTERACTION_root/
├── maps/
├── test_conditional-multi-agent/
├── test_multi-agent/
├── train/
│   ├── DR_CHN_Merging_ZS0_train
│   ├── ...
└── val/
    ├── DR_CHN_Merging_ZS0_val
    ├── ...

```

2. Install the map dependency [lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2):
```
pip install lanelet2==1.2.1
```

</p>
</details>


## Training
Data preprocessing may take several hours the first time you run this project. Training on 4 RTX 4090 GPUs, one epoch takes about 90 and 12 minutes for Argoverse and INTERACTION, respectively.
```
# For Argoverse
python Argoverse/train.py --root /path/to/Argoverse_root/ --train_batch_size 1 --val_batch_size 1 --devices 4

# For INTERACTION
python INTERACTION/train.py --root /path/to/INTERACTION_root/ --train_batch_size 4 --val_batch_size 4 --devices 4
```

## Validation
```
# For Argoverse
python Argoverse/val.py --root /path/to/Argoverse_root/ --val_batch_size 2 --devices 8 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python INTERACTION/val.py --root /path/to/INTERACTION_root/ --val_batch_size 4 --devices 8 --ckpt_path /path/to/checkpoint.ckpt
```

## Testing
```
# For Argoverse
python Argoverse/test.py --root /path/to/Argoverse_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python INTERACTION/test.py --root /path/to/INTERACTION_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt
```

## Acknowledgements
We sincerely appreciate [Argoverse](https://github.com/argoverse/argoverse-api), [INTERACTION](https://github.com/interaction-dataset/interaction-dataset),[QCNet](https://github.com/ZikangZhou/QCNet) and [HPNet](git@github.com:XiaolongTang23/HPNet.git)  for their awesome codebases.
