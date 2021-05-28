# Blur-Training with ImageNet 


## Preparations
- Install Python Packages  
```bash
$ pip install -r requirements.txt
```
Or pull and run [docker image][docker-blur-training] (e.g. blur-training:latest) I made for this training.  
- Get ImageNet images & set the path. If you already have ImageNet, set `IMAGENET_PATH` variable in `train_imagenet1000.py`.  
If not, Download the ImageNet dataset from http://www.image-net.org/  
    (Note that the ImageNet images need to be divided in two subdirectories, ``train/`` and ``val/``.)  
    Then set the path.
    
    
## run examples
General usage example:
```bash
$ python -u src/train_imagenet1000.py --arch [ARCHITECTURE NAME] --mode [TRAINING MODE] -n [EXPERIMENT NAME] 
```  

For `train_imagenet1000.py`, you need to use `--exp_name` or `-n` option to define your experiment's name. 
Then the experiment's name is used for managing results under `train-logs/` directory.
`train-logs/` will automatically be created when you run `train_imagenet1000.py`.   
You can choose the training mode from:   
`normal, all, mix, reversed-single-step, single-step, multi-steps`  
by using `--mode [TRAINING MODE]` option.

- **normal**  
This mode trains Normal model (default: AlexNet).  
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode normal -e 60 -b 64 --lr 0.01 -n alexnet_normal
```
You can also specify GPU by `--gpu`.  
**(You don't need it if you have only single GPU in your machine.)**
```bash
$ python -u src/train_imagenet1000.py --gpu 0 -a alexnet --mode normal -n alexnet_normal
```

- **all**  
This mode blurs ALL images in the training mode.  
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode all -s1 -n alexnet_all_s1
```

- **mix**    
This mode blurs half training data.
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode mix -s 1 -n alexnet_mix_s1
```

- **random-mix** <br>
This mode blurs half training data **randomly**. <br>
usage example:
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode random-mix --min_sigma 0 --max_sigma 5 -n alexnet_random-mix_s0-5
```

- **single-step**    
This mode blurs first half epochs (e.g. first 30 epochs in 60 entire epochs) in the training.
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode single-step -s 1 -n alexnet_single-step_s1
```

- **fixed-single-step**    
This mode blurs first half epochs in the training, then fixes the weights of 1st Conv layer.
usage example:  
```bash
python -u src/train_imagenet1000.py -a alexnet --mode fixed-single-step -s 1 -n alexnet_fixed-single-step_s1
```

- **reversed-single-step**    
This mode is reversed order of single-step (blurs second half epochs).
usage example:  
```bash
$ python -u src/train_imagenet1000.py -a alexnet --mode reversed-single-step --reverse_sigma 1 -n alexnet_reversed-single-step_s1
```

- **multi-steps**  
This mode blurs images step by step (e.g. every 10 epochs).  
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode multi-steps -n alexnet_multi-steps
```

- `--blur_val`   
This option blurs validation data as well. 
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode mix -s 1 --blur_val -n alexnet_mix_lur-val_s1
```

- `--resume [PATH TO SAVED MODEL]`   
This option trains your saved model starting from the latest epoch.  
usage example:  
```bash
$ python -u src/train_imagenet1000.py --arch alexnet --mode mix -s 1 -n alexnet_mix_s1 --resume ../train-logs/models/alexnet_mix_s1/checkpoint.pth.tar 
```

### Multi GPUs
If you want to use multi GPUs trainig, see [pytorch imagenet training example][pytorch-imagenet].
e.g.:
```bash 
$ python -u src/train_imagenet1000.py -a resnet50 --seed 42 --lr 0.2 --mode normal --epochs 60 -b 512 --dist_url 'tcp://127.0.0.1:10000' --dist_backend 'nccl' --multiprocessing_distributed --world_size 1 --rank 0 -n resnet50_normal_b512
```

## citations
Training scripts and functions are based on [pytorch tutorial][pytorch-tutorial] and [pytorch imagenet training example][pytorch-imagenet].



[pytorch-tutorial]:https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
[pytorch-imagenet]:https://github.com/pytorch/examples/blob/master/imagenet
[docker-blur-training]:https://hub.docker.com/r/sousquared/blur-training
