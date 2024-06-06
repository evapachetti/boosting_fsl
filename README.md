

# Boosting Few-Shot Learning with Disentangled Self-Supervised Learning and Meta-Learning for Medical Image Classification [ In progress...] 
Official code for [**Boosting Few-Shot Learning with Disentangled Self-Supervised Learning and Meta-Learning for Medical Image Classification**](https://arxiv.org/abs/2403.17530). The code is based on [DeepBDC](https://github.com/Fei-Long121/DeepBDC) by [FeiLong](https://github.com/Fei-Long121), pytorch code of [Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_Joint_Distribution_Matters_Deep_Brownian_Distance_Covariance_for_Few-Shot_Classification_CVPR_2022_paper.pdf) and on [IP-IRM](https://github.com/Wangt-CN/IP-IRM) by [Wangt-CN](https://github.com/Wangt-CN), pytorch implementation of [Self-Supervised Learning Disentangled Group Representation as Feature](https://proceedings.neurips.cc/paper/2021/file/97416ac0f58056947e2eb5d5d253d4f2-Paper.pdf).

![boosting_fsl](./img/boosting_fsl.png)

## Dataset
We utilized the [PI-CAI](https://zenodo.org/records/6517398) and [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) datasets for our experiments. To see pre-processing details, please refer to our [paper](https://arxiv.org/abs/2403.17530).
Based on our code, the data should be organized according to the following structure:
```
├── dataset
│   └── picai
│       ├── supervised                            
│       ├── unsupervised
│       ├── csv_files
│   └── breakhis
│       ├── supervised                            
│       ├── unsupervised
│       ├── csv_files

```
Here, *supervised* contains the samples used for supervised training, *unsupervised* the samples for the unsupervised pre-training steps, and *csv_files* the CSV files from which to retrieve the sample metadata.
## Citation

```bibtex
@article{pachetti2024boosting,
  title={Boosting Few-Shot Learning with Disentangled Self-Supervised Learning and Meta-Learning for Medical Image Classification},
  author={Pachetti, Eva and Tsaftaris, Sotirios A and Colantonio, Sara},
  journal={arXiv preprint arXiv:2403.17530},
  year={2024}
}
```
