# PyTorch LapSRN
Implementation of TPAMI 2018 Paper: "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks"(http://vllab.ucmerced.edu/wlai24/LapSRN/) in PyTorch.

Code for the evaluation, implementation of the loss and the bicubic filter weights are taken from https://github.com/twtygqyy/pytorch-LapSRN.

As far as i know this is the only Pytorch implementation that isn't based on the 2017 CVPR paper but rather on the paper updates published in 2018.

In particular the 2018 implementation uses weight sharing and skip connection to dramatically reduce the number of learnable parameters while maintaining high performances!

## Usage
### Training

### Evaluation
```
usage: eval_mat.py
```

### Demo
```
usage: demo.py
```

### Prepare Training dataset
  - Download training and testing dataset from https://github.com/phoenix104104/LapSRN/blob/master/datasets/datasets_download_links.txt

### Performance
- Using the parameters contained in the train.py script i was able to achieve these performances using the D5R5 network.
  
| DataSet/Method        | LapSRN Paper          | LapSRN PyTorch|
| ------------- |:-------------:| -----:|
| Set5 (2x)     | **37.72**      | 37.46 |
| Set14 (2x)    | **33.24**      | 33.07 |
| BSD100 (2x)   | **32.00**      | 31.79 |
| Set5 (4x)     | 31.74      | **31.77** |
| Set14 (4x)    | 28.25      | **28.35** |
| BSD100 (4x)   | **27.42**      | 27.41 |


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
