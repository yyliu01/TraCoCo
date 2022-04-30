# TraCoCo
[Translation Consistent Semi-supervised Segmentation for 3D Medical Images](http://arxiv.org/abs/2203.14523)

### Installation
Please install the dependencies and dataset based on the [installation](./docs/installation.md) document.

### Getting start
Please follow this [instruction](./docs/before_start.md) document to reproduce our results.

### Results on the Left Atrium dataset
* The training set consists of 16 labelled scans and 64 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|Reference|
  |---|---|---|---|---|---|
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|88.88|80.21|2.26|7.32|MICCAI2019|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|89.54|81.24|2.20|8.24|MICCAI2020|
  |[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|89.62|81.31| 2.06| 7.16|MICCAI2020|
  |[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|89.65| 81.35| 2.03| 7.04|MICCAI2020|
  |[DTC](https://ojs.aaai.org/index.php/AAAI/article/view/17066)|89.42|80.98|2.10|7.32|AAAI2021|
  |[MC-Net](https://arxiv.org/pdf/2103.02911.pdf)|90.34| 82.48| 1.77| 6.00|MICCAI2021|
  |Ours [(training log)](https://1drv.ms/t/s!AsvBenvUFxO3hmYgPKEa2_w4F5BC?e=LPCIHZ) |90.94| 83.47| 1.79| 5.49|Arxiv|
* The training set consists of 8 labelled scans and 72 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|Reference|
  |---|---|---|---|---|---|
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|84.25|73.48|3.36|13.84|MICCAI2019|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|87.32|77.72|2.55|9.62|MICCAI2020|
  |[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|85.54|75.12|3.77|13.29|MICCAI2020|
  |[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|85.91|75.75|3.31|12.67|MICCAI2020|
  |[DTC](https://ojs.aaai.org/index.php/AAAI/article/view/17066)|87.51|78.17|2.36|8.23|AAAI2021|
  |[MC-Net](https://arxiv.org/pdf/2103.02911.pdf)|87.71|78.31|2.18| 9.36|MICCAI2021|
  |Ours [(training log)](https://1drv.ms/t/s!AsvBenvUFxO3hmUIHM9ntFPoqDbw?e=efvg3h) |89.29| 80.82| 2.28| 6.92|Arxiv|

Note: parts of the tables are borrowed from [here](https://github.com/HiLab-git/DTC/blob/master/README.md), and our train/val sample index follows [UA-MT](https://github.com/yulequan/UA-MT). Our fully training details (i.e., checkpoints and training logs) are available in this [one-drive](https://1drv.ms/u/s!AsvBenvUFxO3hn0BLwjDwSG3Q3pE?e=fT0Df9) link.
### Acknowledgement
The code is partially borrowed from [CPS](https://github.com/charlesCXK/TorchSemiSeg) and
[UA-MT](https://github.com/yulequan/UA-MT). Many thanks for their great work. :)
### Citation
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
- [ ] Code for Brain Tumor Segmentation Challenge 2019 (BRaTS19)

