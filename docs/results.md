# Results
**Our fully training details (i.e., checkpoints and training logs) are available in this [google-drive](https://drive.google.com/drive/folders/1pJh_FIDwTo9eus5NwvNhZ5hlLRNOfYPN?usp=sharing) link.**

## (update) ACDC dataset
* The training set consists of 3 labelled volume and 67 unlabelled scans and the testing set includes 40 volumes.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|
  |[PS-MT](https://arxiv.org/abs/2111.12903) |86.94| 77.90| 2.18| 4.65|
  |[BCP](https://arxiv.org/pdf/2305.00673.pdf) |87.59 | 78.67 | 0.67 | 1.90
  |[UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical)|87.61| 78.68| 1.97| 4.13|
  |Ours [(details)](https://drive.google.com/drive/folders/1arJL-uUnJjmgHhGUc5y3EkPYDjE29rVx?usp=sharing) |89.46| 81.51| 0.71| 2.43|

* The training set consists of 7 labelled volume and 63 unlabelled scans and the testing set includes 40 volumes.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
    |---|---|---|---|---|
  |[PS-MT](https://arxiv.org/abs/2111.12903) |88.91| 80.79| 1.83| 4.96|
  |[BCP](https://arxiv.org/pdf/2305.00673.pdf) |88.84 |80.62| 1.17 |3.98|
  |[BCPCaussl](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_CauSSL_Causality-inspired_Semi-supervised_Learning_for_Medical_Image_Segmentation_ICCV_2023_paper.pdf) |89.66| 81.79 |0.93| 3.67|
  |[UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical)|89.92 | 81.97 | 0.98 | 3.75|
  |Ours [(details)](https://drive.google.com/drive/folders/1bp_ogFe4oGLsVXdkZyfkRaDYR05hxSO5?usp=drive_link) |90.44 | 83.01 | 0.42 | 1.41 |

* The training set consists of 14 labelled volume and 56 unlabelled scans and the testing set includes 40 volumes.
  
  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|
  |[PS-MT](https://arxiv.org/abs/2111.12903) |89.94 |81.90 |1.15 |4.01|
  |[BCP](https://arxiv.org/pdf/2305.00673.pdf) |89.52| 81.62| 1.03| 3.69|
  |[BCPCaussl](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_CauSSL_Causality-inspired_Semi-supervised_Learning_for_Medical_Image_Segmentation_ICCV_2023_paper.pdf) |89.99 |82.34 |0.88 |3.60|
  |[UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical)|90.47 | 82.96| 0.99| 2.04|
  |Ours [(details)](https://drive.google.com/drive/folders/1838GN1IPzXg9E_ih9OCMPryc2yWeSzYU?usp=drive_link) |91.24 |84.32 |0.36 |1.29|

## Left Atrium dataset
_* denotes the results are from the best checkpoints' protocol_
* The training set consists of 16 labelled scans and 64 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|88.88|80.21|2.26|7.32|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|89.54|81.24|2.20|8.24|
  |[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|89.62|81.31| 2.06| 7.16|
  |[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|89.65| 81.35| 2.03| 7.04|
  |[DTC](https://ojs.aaai.org/index.php/AAAI/article/view/17066)|89.42|80.98|2.10|7.32|
  |[MC-Net](https://arxiv.org/pdf/2103.02911.pdf)|90.34| 82.48| 1.77| 6.00|
  |Ours [(training log)](https://drive.google.com/file/d/1p57j9uuFRseTFdwLGvtaEOBbE-uoSs2w/view?usp=sharing) |90.94| 83.47| 1.79| 5.49|
  |Ours* [(training log)](https://drive.google.com/file/d/1L0Pkcv6zL75ZLlJxiHgZkELHfKmgRp90/view?usp=sharing) |91.51| 84.40| 1.79| 5.63|

* The training set consists of 8 labelled scans and 72 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
    |---|---|---|---|---|
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|84.25|73.48|3.36|13.84|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|87.32|77.72|2.55|9.62|M
  |[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|85.54|75.12|3.77|13.29|
  |[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|85.91|75.75|3.31|12.67|M
  |[DTC](https://ojs.aaai.org/index.php/AAAI/article/view/17066)|87.51|78.17|2.36|8.23|A
  |[MC-Net](https://arxiv.org/pdf/2103.02911.pdf)|87.71|78.31|2.18| 9.36|
  |Ours [(training log)](https://drive.google.com/file/d/1h3OKEdURZ4eZoomhoDQZO9ldDg6ZmfLp/view?usp=sharing) |89.29| 80.82| 2.28| 6.92|
  |Ours* [(training log)](https://drive.google.com/file/d/1O5yNaUs_2-QQ6EJ10Cvwfu7-RTEY-atR/view?usp=sharing) |89.86| 81.70| 2.01| 6.81|
  
* _Parts of the tables are borrowed from [here](https://github.com/HiLab-git/DTC/blob/master/README.md), and our train/val sample index follows [UA-MT](https://github.com/yulequan/UA-MT/tree/master/data)._

## Pancreas dataset
_* denotes the results are from the best checkpoints' protocol_

* **(update)** Following [BCP](https://github.com/DeepMed-Lab-ECNU/BCP), we have established our methods based on [best evaluation protocol](https://github.com/DeepMed-Lab-ECNU/BCP/blob/a925e3018b23255e65a62dd34ae9ac9fc18c0bc9/code/pancreas/train_pancreas.py#L134), and the results are shown as below:

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
    |---|---|---|---|---|  
  |[CoraNet](https://arxiv.org/pdf/2305.00673.pdf)|79.67|66.69|1.89|7.59|
  |[BCP](https://arxiv.org/pdf/2305.00673.pdf)|82.91|70.97|2.25|6.43|
  |Ours* [(training log)](https://drive.google.com/file/d/1bBjCvWrbbVO0dhjHTTlDgflJdhhy0_yg/view?usp=drive_link) |83.36| 71.70| 1.74| 7.34|

* The training set consists of 12 labelled scans and 50 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|  
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|76.10|62.62|2.43|10.84|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|76.39|63.17|1.42|11.06|
  |[URPC](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001645)|80.02|67.30|1.98|8.51|
  |[MC-Net+](https://arxiv.org/pdf/2103.02911.pdf)|80.59|68.08|1.74|6.47|
  |Ours [(training log)](https://drive.google.com/file/d/1-VO9T5WI8UR_1H1LIsaU2EKbM6RwURZd/view?usp=sharing) |81.80| 69.56| 1.49|5.70|
  
* The training set consists of 6 labelled scans and 56 unlabelled scans and the testing set includes 20 scans.
  
  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|  
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|66.44|52.02|3.03|17.04|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|68.97|54.29|1.96|18.83|
  |[URPC](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001645)|73.53|59.44|7.85| 22.57|
  |[MC-Net+](https://linkinghub.elsevier.com/retrieve/pii/S1361841522001773)|74.01| 60.02| 3.34| 12.59|
  |Ours [(training log)](https://drive.google.com/file/d/1u1lqevL1ZTDGRxg0TSFjaJZaoYs_p8p9/view?usp=drive_link) |79.22| 66.04| 2.57| 8.46|

* _Our train/val sample index follows [MC-Net+](https://github.com/ycwu1997/MC-Net/tree/main/data/Pancreas)._

## BRaTS19 dataset
* The training set consists of 50 labelled scans and 200 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|  
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|85.32| 75.93| 1.98| 8.68 |
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|85.64 |76.33| 2.04| 9.17 |
  |[URPC](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001645)|85.38 |76.14| 1.87 |8.36 |
  |[MC-Net+](https://arxiv.org/pdf/2103.02911.pdf)|86.02| 76.98| 1.98| 8.74 |
  |Ours [(training log)](https://drive.google.com/file/d/1cLOueKfFTAm_YUbmLu5WJq_Nm10ATBRq/view?usp=share_link) |86.69| 77.69| 1.93| 8.04|

* The training set consists of 25 labelled scans and 225 unlabelled scans and the testing set includes 20 scans.

  |Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|
  |---|---|---|---|---|  
  |[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|84.64 |74.76 |2.36 |10.47|
  |[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|84.73| 74.89| 2.44| 9.88|
  |[URPC](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001645)|84.53| 74.60| 2.55| 9.79|
  |[MC-Net+](https://linkinghub.elsevier.com/retrieve/pii/S1361841522001773)|84.96| 75.14| 2.36| 9.45 |
  |Ours [(training log)](https://drive.google.com/file/d/1eVvdpYlQHU-XgV34WmwQcRwUgrLU1cuE/view?usp=share_link) |85.71| 76.39| 2.27 |9.20 |
-----
