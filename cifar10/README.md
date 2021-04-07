# Blur-Training\_Cifar10

## Preparations
If you want to run on GPU, you need to check your cuda's version and install pytorch like below.
```bash
$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
(Check [pytorh homepage][pytorch-hp] for more information.)     
Then install Python Packages  
```bash
$ pip install -r requirements.txt
```
<br/>

OR pull and run [docker image][docker-blur-training] (e.g. blur-training:1.0) I made for this experiments.  


## Architecture
**AlexNet-Cifar10**  
This AlexNet has different kernel-size and dense-size due to the image size of Cifar10. This AlexNet is the same structure with [this site (in Japanese)][alexnet-cifar10].


## Usage: ./training/main.py
General usage example:
```bash
$ cd ./training
$ python main.py --mode [TRAINING MODE] -n [EXPERIMENT NAME]
```  

For `main.py`, you need to use `--exp-name` or `-n` option to define your experiment's name. Then the experiment's name is used for managing results under `logs/` directory.   
You can choose the training mode from {normal,blur-all,blur-half-epochs,blur-step,blur-half-data} by using `--mode [TRAINING MODE]` option.

- **normal**  
This mode trains Normal alexnetCifar10.  
usage example:  
```bash
$ python main.py --mode normal -e 60 -n normal_60e
```

- **blur-all**  
This mode blurs ALL images in the training mode.  
usage exmaple:  
```bash
$ python main.py --mode blur-all -s 1 -n blur-all_s1
```

- **blur-half-epochs**    
This mode blurs first half epochs (e.g. first 30 epochs in 60 entire epochs) in the training.
usage example:  
```bash
$ python main.py --mode blur-half-epochs -s 1 -n blur-half-epochs_s1
```

- **blur-half-data**    
This mode blurs half training data.
usage example:  
```bash
$ python main.py --mode blur-half-data -s 1 -n blur-half-data_s1
```

- **blur-step**  
This mode blurs images step by step (e.g. every 10 epochs).  
usage example:  
```bash
$ python main.py --mode blur-step -n blur-step
```

- `--blur-val`   
This option blurs validation data as well. 
usage example:  
```bash
$ python main.py --mode blur-half-epochs -s 1 --blur-val -n blur-half-epochs_blur-val_s1
```

- `--resume [PATH TO SAVED MODEL]`   
This option trains Normal alexnetCifar10 from your saved model.  
usage example:  
```bash
python main.py -e 90 --mode normal --resume ../logs/models/blur-half-epochs_s1/model_060.pth.tar -n blur-half-epochs_s1_from60e
```


## logs/

`logs/` directory will automaticaly be created when you run one of training scripts.  
`logs/` directory contains `outputs/`, `models/`, and `tb/` directories.  

- `logs/outputs/` : records "stdout" and "stderr" from the training scripts.
- `logs/models/` : records model parameters in the form of pytorch state (default: every 10 epochs). 
- `logs/tb/` : records tensorboard outputs. (acc/train, acc/val, loss/train, loss/val)


## dataset/: Cifar10
`dataset/` directory will automatically be created when you run one of training scripts.  


## notebooks/  
Demonstrations and examples of Gaussian Blur.  


## citations
Training scripts and functions are strongly relied on [pytorch tutorial][pytorch-tutorial] and [pytorch imagenet trainning example][pytorch-imagenet].


[alexnet-cifar10]:http://cedro3.com/ai/pytorch-alexnet/
[pytorch-tutorial]:https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
[pytorch-imagenet]:https://github.com/pytorch/examples/blob/master/imagenet/main.py
[docker-blur-training]:https://hub.docker.com/r/sousquared/blur-training
[pytorch-hp]:https://pytorch.org/
