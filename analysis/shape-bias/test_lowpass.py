#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re


import numpy as np
import pandas as pd
import torch

from vonenet import get_model
from src.model.load_sin_pretrained_models import load_sin_model
# from https://github.com/rgeirhos/texture-vs-shape/tree/master/code
import probabilities_to_decision

from utils import label_map, get_key_from_value, load_model, make_dataloader

sys.path.append('../../')
from training.utils import GaussianBlurAll


arch = sys.argv[1]
epoch = 60
MODELS_DIR = '../../logs/models/'  # model directory
RESULTS_DIR = './results/{}_lowpass'.format(arch)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)


# create mapping module
mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()    
def test(model_name, arch, epoch, test_sigma): 
    """Test with cue-conflict images and record correct decisions.
    * You need to exclude images without a cue conflict (e.g. texture=cat, shape=cat)
    Dataset: Cue-conflict
        Size: 224 x 224 (RGB)
        Dataset size: 1,200 images
            # The number of total images is 1,280.
            # The number of images without a cue conflict is 80 (5 per each shape category)
            # Thus, the number of the images becomes 1,200.
            # The number of shape images: 75 (per each shape category)
            # The number of texture images: 75 (per each tecture category)
    """
    num_classes = 16  # number of classes
    model_path = os.path.join(MODELS_DIR, model_name, 'epoch_{}.pth.tar'.format(epoch))
    
    # load model
    if 'vonenet' in model_name:
        model = get_model(model_arch=arch, pretrained=True).to(device)
    elif 'SIN' in model_name:
        model = load_sin_model(model_name).to(device)
    else:
        model = load_model(model_path, arch).to(device)
    print(model_name, test_sigma)
    
    # === compute shape-vs-texture decisions ===
    # make numpy arrays for recording
    correct_shape_decisions = np.zeros(num_classes)
    correct_texture_decisions = np.zeros(num_classes)
    all_results = []
    all_file_names = []
    # make dataloader
    cue_conf_loader = make_dataloader()  
    model.eval()
    with torch.no_grad():
        for images, _, file_names in cue_conf_loader:
            all_file_names.extend(file_names)
            # blur test images
            images = GaussianBlurAll(images, sigma=test_sigma)
            images = images.to(device)
            # get labels from file names
            # first one is shape label. second one is texture label
            labels = [re.sub('\d+', '', f.split('.')[0]).split('-') for f in file_names]
            outputs = model(images)
            outputs = torch.nn.Softmax(dim=1)(outputs)  # sofmax
            
            for i in range(outputs.shape[0]):
                # get label keys from label_map (type: integer)
                shape_id, texture_id = [get_key_from_value(label_map, v) for v in labels[i]]
                # get model_decision (str) by mappig outputs from 1,000 to 16
                model_decision = mapping.probabilities_to_decision(outputs[i].detach().cpu().numpy())
                # record all results
                all_results.append([
                                label_map[shape_id],  # shape label
                                label_map[texture_id],  # texture label
                                model_decision,  # model decision
                                # *outputs[i].cpu().detach().numpy()  # all outputs
                                ])
                
                if not shape_id == texture_id:  # Exclude images without a cue conflict (e.g. texture=cat, shape=cat)
                    # Sum up shape or texture decisions (when either shape or texture category is correctly predicted).
                    correct_shape_decisions[shape_id] += float((model_decision == label_map[shape_id])) 
                    correct_texture_decisions[texture_id] += float((model_decision == label_map[texture_id]))  
    
    # save all decisions
    df_all_decisions = pd.DataFrame(all_results, index=all_file_names, 
                                columns=np.array(['shape_label', 'texture_label', 'model_decision', 
                                                 # *label_map.values()
                                                 ]))
    if epoch == 0:
        filename = 'all_decisions_{}_test-s{}.csv'.format(model_name, test_sigma)
    else:
         filename = 'all_decisions_{}_e{}_test-s{}.csv'.format(model_name, epoch, test_sigma)
    df_all_decisions.to_csv(os.path.join(RESULTS_DIR, filename))
    
    # save correct decisions
    correct_results = np.concatenate([correct_shape_decisions.reshape(1, -1), correct_texture_decisions.reshape(1, -1)])
    df_correct_decisions = pd.DataFrame(correct_results, 
                                        index=['correct_shape_decisions', 'correct_texture_decisions'], 
                                        columns=list(label_map.values()))
    if epoch == 0:
        filename = 'correct_decisions_{}_test-s{}.csv'.format(model_name, test_sigma)
    else:
        filename = 'correct_decisions_{}_e{}_test-s{}.csv'.format(model_name, epoch, test_sigma)
    df_correct_decisions.to_csv(os.path.join(RESULTS_DIR, filename))
    

model_names = {
    'alexnet': "alexnet_trained_on_SIN", 
    'vgg16': "vgg16_trained_on_SIN",
    'resnet50': "resnet50_trained_on_SIN"
}
for test_sigma in range(51):
    # normal
    model_name = '{}_normal'.format(arch)
    test(model_name, arch, epoch, test_sigma)

    # all
    for s in range(1, 5):
        model_name = '{}_all_s{}'.format(arch, s)
        test(model_name, arch, epoch, test_sigma)

    # mix
    for s in range(1,5):
        model_name = '{}_mix_s{}_b512'.format(arch, s)
        test(model_name, arch, epoch, test_sigma)

    # single-step
    for s in range(1,5):
        model_name = '{}_single-step_s{}_b512'.format(arch, s)
        test(model_name, arch, epoch, test_sigma)
    if arch == 'alexnet':
        # fixed-single-step
        for s in range(1,5):
            model_name = '{}_fixed-single-step_s{}'.format(arch, s)
            test(model_name, arch, epoch, test_sigma)
    
    # reversed-single-step
    # for s in range(1, 5):
    #     model_name = '{}_reversed-single-step_s{}'.format(arch, s)
    #     test(model_name, arch, epoch, test_sigma)

    # multi-steps
    model_name = '{}_multi-steps'.format(arch)
    test(model_name, arch, epoch, test_sigma)
    
    # VOneNet
    model_name = '{}_vonenet'.format(arch)
    test(model_name, arch, epoch=0, test_sigma=test_sigma)
    
    # Stylized-ImageNet
    test(model_names[arch], arch, epoch=0, test_sigma=test_sigma)