# Installation
The project is based on the pytorch 1.9.0 with python 3.7. 

We use a 3090 to train the LA and a 32G V100 to train the BRaTS19.
### 1. Clone the Git  repo
``` shell
$ git clone https://github.com/yyliu01/TraCoCo.git
$ cd TraCoCo
```
### 2. Install dependencies
1) create conda env
    ```shell
    $ conda env create -f tracoco.yml
    ```
2) install the torch 1.9.0 
    ```shell
    $ conda activate tracoco
    # IF cuda version < 11.0
    $ pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
    # IF cuda version >= 11.0 (e.g., 30x or above)
    $ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
   
### 3. Prepare dataset

1) download the LA, Pancreas and BRaTs19 datasets from this [google drive link](https://drive.google.com/drive/folders/1_cOBDNlMYjNG-CJFzbmWXXGPj2vvKESE?usp=sharing).
2) organize the filepath as the tree struct shown below.
```
TraCoCo/
├── Code
│   ├── UnetBRATS
│   ├── VnetLA
│   └── VnetPancreas
├── Datasets
│   ├── BRATS19
│   ├── Left_Atrium
│   └── Pancreas
├── docs
├── k8s_launch
│   └── jobs
└── scripts
```

