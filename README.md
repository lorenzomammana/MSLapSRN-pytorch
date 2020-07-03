# PyTorch LapSRN implementation with weight sharing and skip connections (MSLapSRN)

Implementation of TPAMI 2018 Paper: "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks"(http://vllab.ucmerced.edu/wlai24/LapSRN/) in PyTorch.

Code for the evaluation, implementation of the loss and the bicubic filter weights are taken from https://github.com/twtygqyy/pytorch-LapSRN.

As far as i know this is the only Pytorch implementation that isn't based on the 2017 CVPR paper but rather on the paper updates published in 2018.

In particular the 2018 implementation uses weight sharing and skip connection to dramatically reduce the number of learnable parameters while maintaining high performances!

## Usage
### Training
```
usage: train.py
```
### Evaluation
```
usage: eval_mat.py
```

### Demo
```
usage: demo.py
```

### Prepare Datasets
  - Download training and testing dataset from https://github.com/phoenix104104/LapSRN/blob/master/datasets/datasets_download_links.txt
  - Create directory dataset
  - Create subdirectories train, test, validation
  - Place the BSDS200 + T91 images inside the train folder
  - Place the General100 images inside the validation folder
  - Place the Set5, Set14, BSDS100 folders inside the test folder
 
### Prepare evaluation
  - Create a folder named mat inside dataset
  - For each test set create a subfolder
  - Inside each subfolder create a 2x folder and a 4x folder
  - Launch the matlab script
```
usage: generate_test_mat.m
```
This matlab script will produce one .mat file for all the testing files containg the 2x and 4x images resized following the author previous work.

### Folder structure
The final folder structure is going to be something like this
```bash
├── MSLapSRN-pytorch
│   ├── dataset
│   │   ├── mat
│   │   │   ├── Set5
│   │   │   │   ├── 2x
│   │   │   │   ├── 4x
│   │   │   ├── Another test set following the same schema
│   │   ├── test
│   │   │   ├── test set folders containing images
│   │   ├── train
│   │   │   ├── 291 image
│   │   ├── validation
│   │   │   ├── 100 images
├── *.py
├── README.md
└── .gitignore
```
### Performance
- Using the parameters contained in the train.py script i was able to achieve these performances using the D5R5 network.
  
| DataSet/Method        | LapSRN Paper (PSNR)          | LapSRN PyTorch (PSNR) | LapSRN Paper (SSIM) | LapSRN PyTorch (SSIM) | 
| ------------- |:-------------:| :-------------: |:-------------:| :-------------:|
| Set5 (2x)     | **37.72**      | 37.49 | **0.960**      | 0.958 | 
| Set14 (2x)    | **33.24**      | 33.11 | **0.914**      | 0.912 | 
| BSD100 (2x)   | **32.00**      | 31.83 | **0.898**      | 0.893 | 
| Set5 (4x)     | 31.74      | **31.83** | 0.888      | **0.890** | 
| Set14 (4x)    | 28.25      | **28.39** | 0.773    | **0.775** | 
| BSD100 (4x)   | **27.42**      | **27.42** | **0.731** | 0.729 | 

I've tested only the D5R5 implementation with 2x, 4x scaling, as the network is dynamically generated it's easy to increase the scale or change the recursion and depth!

### Citation

If you find the code and datasets useful in your research, please consider citing:
    
    @article{DBLP:journals/corr/abs-1710-01992,
      author    = {Wei{-}Sheng Lai and
                   Jia{-}Bin Huang and
                   Narendra Ahuja and
                   Ming{-}Hsuan Yang},
      title     = {Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid
                   Networks},
      journal   = {CoRR},
      volume    = {abs/1710.01992},
      year      = {2017},
    }
