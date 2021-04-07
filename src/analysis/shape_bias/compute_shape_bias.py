import os
import sys
import re

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from utils import (
    label_map,
    get_key_from_value,
    ImageFolderWithFileName,
    load_model,
    AverageMeter,
    accuracy,
    make_dataloader,
)


arch = sys.argv[1]
epoch = 60
MODELS_DIR = "../../logs/models/"  # model directory
RESULTS_DIR = "./results/{}".format(arch)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


def test(model_name, arch, epoch):
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
    model_path = os.path.join(MODELS_DIR, model_name, "epoch_{}.pth.tar".format(epoch))
    save_path = os.path.join(
        RESULTS_DIR, "all_decisions_{}_e{}.csv".format(model_name, epoch)
    )  # save path of the result

    # load model
    if "vonenet" in model_name:
        model = get_model(model_arch=arch, pretrained=True).to(device)
    elif "SIN" in model_name:
        model = load_sin_model(model_name).to(device)
    else:
        model = load_model(model_path, arch).to(device)
    print(model_name)

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
            images = images.to(device)
            # get labels from file names
            # first one is shape label. second one is texture label
            labels = [re.sub("\d+", "", f.split(".")[0]).split("-") for f in file_names]
            outputs = model(images)

            for i in range(outputs.shape[0]):
                # get label keys from label_map (type: integer)
                shape, texture = [get_key_from_value(label_map, v) for v in labels[i]]
                _, pred = outputs[i].topk(1)
                # record all results
                all_results.append(
                    [
                        label_map[shape],  # shape label
                        label_map[texture],  # texture label
                        label_map[pred.item()],  # model decision
                        *outputs[i].cpu().detach().numpy(),  # all outputs
                    ]
                )

                if (
                    not shape == texture
                ):  # Exclude images without a cue conflict (e.g. texture=cat, shape=cat)
                    # Sum up shape or texture decisions (when either shape or texture category is correctly predicted).
                    correct_shape_decisions[shape] += pred.eq(
                        shape
                    ).float()  # sum up when the correct decision is shape
                    correct_texture_decisions[texture] += pred.eq(
                        texture
                    ).float()  # sum up when the correct decision is texture

    # save all decisions
    df_all_decisions = pd.DataFrame(
        all_results,
        index=all_file_names,
        columns=np.array(
            ["shape_label", "texture_label", "model_decision", *label_map.values()]
        ),
    )
    if epoch == 0:
        filename = "all_decisions_{}.csv".format(model_name)
    else:
        filename = "all_decisions_{}_e{}.csv".format(model_name, epoch)
    df_all_decisions.to_csv(os.path.join(RESULTS_DIR, filename))

    # save correct decisions
    correct_results = np.concatenate(
        [
            correct_shape_decisions.reshape(1, -1),
            correct_texture_decisions.reshape(1, -1),
        ]
    )
    df_correct_decisions = pd.DataFrame(
        correct_results,
        index=["correct_shape_decisions", "correct_texture_decisions"],
        columns=list(label_map.values()),
    )
    if epoch == 0:
        filename = "correct_decisions_{}.csv".format(model_name)
    else:
        filename = "correct_decisions_{}_e{}.csv".format(model_name, epoch)
    df_correct_decisions.to_csv(os.path.join(RESULTS_DIR, filename))


if __name__ == "__main__":
    # normal
    model_name = "{}_normal".format(arch)
    test(model_name, arch, epoch)

    # # all
    # for s in range(1, 5):
    #     model_name = '{}_all_s{:02d}'.format(arch, s)
    #     test(model_name, arch, epoch)

    #     # mix
    #     sigmas_mix = [s for s in range(1, 6)] + [10]
    #     for s in sigmas_mix:
    #         model_name = "{}_mix_s{:02d}".format(arch, s)
    #         test(model_name, arch, epoch)

    #     # random-mix
    #     sigmas_random_mix = ["00-05", "00-10"]
    #     for s in sigmas_random_mix:
    #         model_name = "{}_random-mix_s{}".format(arch, s)
    #         test(model_name, arch, epoch)

    # # single-step
    # for s in range(1,5):
    #     model_name = '{}_single-step_s{:02d}'.format(arch, s)
    #     test(model_name, arch, epoch)
    # """
    # # fixed-single-step
    # for s in range(1,5):
    #     model_name = '{}_fixed-single-step_s{:02d}'.format(arch, s)
    #     test(model_name, arch, epoch)
    # """
    # # reversed-single-step
    # for s in range(1, 5):
    #     model_name = '{}_reversed-single-step_s{:02d}'.format(arch, s)
    #     test(model_name, arch, epoch)

    # # multi-steps
    # model_name = '{}_multi-steps'.format(arch)
    # test(model_name, arch, epoch)
