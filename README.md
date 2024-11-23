# TraCoCo
### Translation Consistent Semi-supervised Segmentation for 3D Medical Images
>
> by Yuyuan Liu, [Yu Tian](https://yutianyt.com/), Chong Wang, Yuanhong Chen, Fengbei Liu,
> [Vasileios Belagiannis](https://campar.in.tum.de/Main/VasileiosBelagiannis) and [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/)


<img src="https://user-images.githubusercontent.com/102338056/233821073-0484586c-31a0-4f3f-b4e4-e4837bb2427a.png" width="800" height="300" />

## update
  * :rocket: results of best evaluation protocol in Pancreas-CT (including checkpoints and training logs) are available now.
  * :sparkles: results for ACDC benchmark (including codes, checkpoints and training logs) are available now.


### Installation
Please install the dependencies and dataset based on the [installation](./docs/installation.md) document.

### Getting start
Please follow this [instruction](./docs/before_start.md) document to reproduce our results.

### Results
Our training logs and checkpoints are in this [result](./docs/results.md) page.

### Acknowledgement
The code is partially borrowed from [CPS](https://github.com/charlesCXK/TorchSemiSeg) and
[UA-MT](https://github.com/yulequan/UA-MT). Many thanks for their great work. :)

### Citation
Please consider citing our paper in your publications if it helps your research.
```bibtex
@article{liu2022translation,
  title={Translation Consistent Semi-supervised Segmentation for 3D Medical Images},
  author={Liu, Yuyuan and Tian, Yu and Wang, Chong and Chen, Yuanhong and Liu, Fengbei and Belagiannis, Vasileios and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2203.14523},
  year={2022}
}
```

#### TODO
- [x] Code for Left Atrium (LA)
- [x] Code for Pancreas-CT (Pancreas)
- [x] Code for Brain Tumor Segmentation Challenge 2019 (BraTS19)
- [x] Code for Automated Cardiac Diagnosis Challenge (ACDC) 
