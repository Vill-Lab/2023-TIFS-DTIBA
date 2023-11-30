# Invisible backdoor attack with dynamic triggers against person re-identification (IEEE T-IFS 2023)
This is the pytorch implementation of the [paper](https://ieeexplore.ieee.org/abstract/document/10285514/) (accepted by IEEE T-IFS 2023). 

## Getting Started with DT-IBA

1. **Installation:**
   - First, install the required packages listed in `requirements.txt` by running the following command in the command line:
     ```
     pip install -r requirements.txt
     ```

2. **Prepare Pretrained Model:**
   - The feature extraction phase of our identity hashing network uses a pre-trained SBS-R101. You can obtain it from FastReID's GitHub repository [here](https://github.com/JDAI-CV/fast-reid/).
   - The steganography network is referenced from StegaStamp and needs to be retrained according to the image size of the dataset. You can find the StegaStamp GitHub repository [here](https://github.com/tancik/StegaStamp/).
   
   References:
   - FastReID: [A Pytorch Toolbox for General Instance Re-identification](https://arxiv.org/abs/2006.02631)
     ```
     @article{he2020fastreid,
       title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
       author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
       journal={arXiv preprint arXiv:2006.02631},
       year={2020}
     }
     ```
   - StegaStamp: [Invisible Hyperlinks in Physical Photographs](https://arxiv.org/abs/1912.11099)
     ```
     @inproceedings{2019stegastamp,
         title={StegaStamp: Invisible Hyperlinks in Physical Photographs},
         author={Tancik, Matthew and Mildenhall, Ben and Ng, Ren},
         booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
         year={2020}
     }
     ```

3. **Dataset Preparation:**
   - Put the dataset to be poisoned into `./dataset`.
   - Set the input and output path, parameters of the pre-trained model, and the configuration of the victim model in `backdoor_implantation.py`.
   - Run the following command:
     ```
     python backdoor_implantation.py
     ```

**Citation:**
- If you use the DT-IBA method in your work, please cite this paper:
  ```
  @ARTICLE{10285514,
  author={Sun, Wenli and Jiang, Xinyang and Dou, Shuguang and Li, Dongsheng and Miao, Duoqian and Deng, Cheng and Zhao, Cairong},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Invisible Backdoor Attack With Dynamic Triggers Against Person Re-Identification}, 
  year={2024},
  volume={19},
  number={},
  pages={307-319},
  doi={10.1109/TIFS.2023.3322659}}

  ```
