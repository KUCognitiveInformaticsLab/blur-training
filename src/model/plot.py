colors = {
    f"raw_images": "black",
    f"untrained_alexnet": "gray",
    f"alexnet_normal": "#1f77b4",
    f"alexnet_multi-steps": "mediumvioletred",
    f"alexnet_all_s04": "darkgreen",
    f"alexnet_mix_s01": "green",
    f"alexnet_mix_s04": "darkorange",
    f"alexnet_mix_s00-04": "darkorange",
    f"alexnet_mix_s10": "hotpink",
    f"alexnet_mix_s01_no-blur-1label": "limegreen",
    f"alexnet_mix_s01_no-blur-8label": "lime",
    f"alexnet_mix_s04_no-blur-1label": "orangered",
    f"alexnet_mix_s04_no-blur-8label": "gold",
    f"alexnet_mix_s04_no-sharp-1label": "orangered",
    f"alexnet_mix_s04_no-sharp-8label": "gold",
    f"alexnet_mix_p-blur_s01": "lime",
    f"alexnet_mix_p-blur_s04": "darkorange",
    f"alexnet_mix_p-blur_s01_no-blur-1label": "limegreen",
    f"alexnet_mix_p-blur_s01_no-blur-8label": "lighteseagreen",
    f"alexnet_mix_p-blur_s04_no-blur-1label": "orangered",
    f"alexnet_mix_p-blur_s04_no-blur-8label": "gold",
    f"alexnet_random-mix_s00-04": "gold",
    f"alexnet_random-mix_s00-05": "green",
    f"alexnet_random-mix_s00-10": "darkgreen",
    f"alexnet_trained_on_SIN": "brown",
    f"vone_alexnet": "navy",
    f"alexnet_vonenet": "deepskyblue",
    f"untrained_vone_alexnet": "gray",
    f"vone_alexnet_normal": "#1f77b4",
    f"vone_alexnet_multi-steps": "mediumvioletred",
    f"vone_alexnet_all_s04": "darkgreen",
    f"vone_alexnet_mix_s01": "green",
    f"vone_alexnet_mix_s04": "darkorange",
    f"vone_alexnet_random-mix_s00-04": "gold",
    f"vone_alexnet_mix_s10": "hotpink",
    f"vone_alexnet_mix_s01_no-blur-1label": "limegreen",
    f"vone_alexnet_mix_s01_no-blur-8label": "lime",
    f"vone_alexnet_mix_s04_no-blur-1label": "orangered",
    f"vone_alexnet_mix_s04_no-blur-8label": "gold",
    f"vone_alexnet_mix_s04_no-sharp-1label": "orangered",
    f"vone_alexnet_mix_s04_no-sharp-8label": "gold",
    # vgg16
    f"untrained_vgg16": "gray", 
    f"vgg16_normal": "#1f77b4",
    f"vgg16_all_s04": "darkgreen",
    f"vgg16_mix_s04": "darkorange",
    f"vgg16_multi-steps": "mediumvioletred",
    # resnet50
    f"untrained_resnet50": "gray", 
    f"resnet50_normal": "#1f77b4",
    f"resnet50_all_s04": "darkgreen",
    f"resnet50_mix_s04": "darkorange",
    f"resnet50_multi-steps": "mediumvioletred",
    # SimCLR
    "resnet50-1x_simclr": "plum",
    "resnet50-2x_simclr": "mediumorchid",
    "resnet50-4x_simclr": "darkviolet",
    # humans
    "humans": "dimgray",  # ("dimgray", "crimson")
}

lines = {
    f"raw_images": "--",
    f"untrained_alexnet": ":",
    f"alexnet_normal": ":",
    f"alexnet_multi-steps": "-",
    f"alexnet_all_s04": "-",
    f"alexnet_mix_s01": "-",
    f"alexnet_mix_s04": "-",
    f"alexnet_mix_s00-04": "-",
    f"alexnet_mix_s10": "-",
    f"alexnet_mix_s01_no-blur-1label": "-",
    f"alexnet_mix_s01_no-blur-8label": "-",
    f"alexnet_mix_s04_no-blur-1label": "-",
    f"alexnet_mix_s04_no-blur-8label": "-",
    f"alexnet_mix_s04_no-sharp-1label": "-",
    f"alexnet_mix_s04_no-sharp-8label": "-",
    f"alexnet_mix_p-blur_s01": "-",
    f"alexnet_mix_p-blur_s04": "-",
    f"alexnet_mix_p-blur_s01_no-blur-1label": "-",
    f"alexnet_mix_p-blur_s01_no-blur-8label": "-",
    f"alexnet_mix_p-blur_s04_no-blur-1label": "-",
    f"alexnet_mix_p-blur_s04_no-blur-8label": "-",
    f"alexnet_random-mix_s00-04": "-",
    f"alexnet_random-mix_s00-05": "-",
    f"alexnet_random-mix_s00-10": "-",
    f"alexnet_trained_on_SIN": "-",
    f"vone_alexnet": "-",
    f"vone_alexnet_normal": "--",
    f"vone_alexnet_multi-steps": "--",
    f"vone_alexnet_all_s04": "--",
    f"vone_alexnet_mix_s01": "--",
    f"vone_alexnet_mix_s04": "--",
    f"vone_alexnet_random-mix_s00-04": "--",
    f"vone_alexnet_mix_s10": "--",
    f"vone_alexnet_mix_s01_no-blur-1label": "--",
    f"vone_alexnet_mix_s01_no-blur-8label": "--",
    f"vone_alexnet_mix_s04_no-blur-1label": "--",
    f"vone_alexnet_mix_s04_no-blur-8label": "--",
    f"vone_alexnet_mix_s04_no-sharp-1label": "--",
    f"vone_alexnet_mix_s04_no-sharp-8label": "--",
    # vgg16
    f"untrained_vgg16": ":",
    f"vgg16_normal": ":",
    f"vgg16_multi-steps": "-",
    f"vgg16_all_s04": "-",
    f"vgg16_mix_s01": "-",
    f"vgg16_mix_s04": "-",
    # resnet50
    f"untrained_resnet50": ":",
    f"resnet50_normal": ":",
    f"resnet50_multi-steps": "-",
    f"resnet50_all_s04": "-",
    f"resnet50_mix_s01": "-",
    f"resnet50_mix_s04": "-",
    # SimCLR
    "resnet50-1x_simclr": "-",
    "resnet50-2x_simclr": "-",
    "resnet50-4x_simclr": "-",
    "humans": "--",
}


def get_marker(model_name: str, num_classes: int=16):
    if model_name == "humans":
        return "x"
    elif "vgg16" in model_name:
        return "P"  # plus (filled)
    elif "resnet50" in model_name:
        return "s"  # square
    elif "vone_" in model_name:
        return "v"  # triangle_down
    else:
        if num_classes == 1000:
            return "^"  # triangle_up
        elif num_classes ==16:
            return "o"  # circle


def get_hatch(model_name, num_classes: int=16):
    if model_name == "humans":
        return None
    if "vgg16" in model_name:
        return "."
    elif "resnet50" in model_name:
        return "x"
    elif "vone_" in model_name:
        return "///"
    else:
        if num_classes == 1000:
            return "o"
        elif num_classes ==16:
            return None


def get_color(arch, model_name):
    colors = {
        f"raw_images": "black",
        f"untrained_alexnet": "gray",
        f"{arch}_normal": "#1f77b4",
        f"{arch}_multi-steps": "mediumvioletred",
        f"{arch}_all_s04": "darkgreen",
        f"{arch}_mix_s01": "green",
        f"{arch}_mix_s04": "darkorange",
        f"{arch}_mix_s00-04": "darkorange",
        f"{arch}_mix_s10": "hotpink",
        f"{arch}_mix_s01_no-blur-1label": "limegreen",
        f"{arch}_mix_s01_no-blur-8label": "lime",
        f"{arch}_mix_s04_no-blur-1label": "orangered",
        f"{arch}_mix_s04_no-blur-8label": "gold",
        f"{arch}_mix_s04_no-sharp-1label": "orangered",
        f"{arch}_mix_s04_no-sharp-8label": "gold",
        f"{arch}_mix_p-blur_s01": "lime",
        f"{arch}_mix_p-blur_s04": "darkorange",
        f"{arch}_mix_p-blur_s01_no-blur-1label": "limegreen",
        f"{arch}_mix_p-blur_s01_no-blur-8label": "lighteseagreen",
        f"{arch}_mix_p-blur_s04_no-blur-1label": "orangered",
        f"{arch}_mix_p-blur_s04_no-blur-8label": "gold",
        f"{arch}_random-mix_s00-04": "gold",
        f"{arch}_random-mix_s00-05": "green",
        f"{arch}_random-mix_s00-10": "darkgreen",
        f"{arch}_trained_on_SIN": "brown",
        f"vone_{arch}": "navy",
        f"{arch}_vonenet": "deepskyblue",
        f"untrained_vone_{arch}": "gray",
        f"vone_{arch}_normal": "#1f77b4",
        f"vone_{arch}_multi-steps": "mediumvioletred",
        f"vone_{arch}_all_s04": "darkgreen",
        f"vone_{arch}_mix_s01": "green",
        f"vone_{arch}_mix_s04": "darkorange",
        f"vone_{arch}_random-mix_s00-04": "gold",
        f"vone_{arch}_mix_s10": "hotpink",
        f"vone_{arch}_mix_s01_no-blur-1label": "limegreen",
        f"vone_{arch}_mix_s01_no-blur-8label": "lime",
        f"vone_{arch}_mix_s04_no-blur-1label": "orangered",
        f"vone_{arch}_mix_s04_no-blur-8label": "gold",
        f"vone_{arch}_mix_s04_no-sharp-1label": "orangered",
        f"vone_{arch}_mix_s04_no-sharp-8label": "gold",
        "resnet50-1x_simclr": "plum",
        "resnet50-2x_simclr": "mediumorchid",
        "resnet50-4x_simclr": "darkviolet",
        "humans": "dimgray",  # ("dimgray", "crimson")
    }
    
    return colors[model_name]

def get_line(arch, model_name):
    lines = {
    f"raw_images": "--",
    f"untrained_{arch}": ":",
    f"{arch}_normal": ":",
    f"{arch}_multi-steps": "-",
    f"{arch}_all_s04": "-",
    f"{arch}_mix_s01": "-",
    f"{arch}_mix_s04": "-",
    f"{arch}_mix_s00-04": "-",
    f"{arch}_mix_s10": "-",
    f"{arch}_mix_s01_no-blur-1label": "-",
    f"{arch}_mix_s01_no-blur-8label": "-",
    f"{arch}_mix_s04_no-blur-1label": "-",
    f"{arch}_mix_s04_no-blur-8label": "-",
    f"{arch}_mix_s04_no-sharp-1label": "-",
    f"{arch}_mix_s04_no-sharp-8label": "-",
    f"{arch}_mix_p-blur_s01": "-",
    f"{arch}_mix_p-blur_s04": "-",
    f"{arch}_mix_p-blur_s01_no-blur-1label": "-",
    f"{arch}_mix_p-blur_s01_no-blur-8label": "-",
    f"{arch}_mix_p-blur_s04_no-blur-1label": "-",
    f"{arch}_mix_p-blur_s04_no-blur-8label": "-",
    f"{arch}_random-mix_s00-04": "-",
    f"{arch}_random-mix_s00-05": "-",
    f"{arch}_random-mix_s00-10": "-",
    f"{arch}_trained_on_SIN": "-",
    f"vone_{arch}": "-",
    f"vone_{arch}_normal": "--",
    f"vone_{arch}_multi-steps": "--",
    f"vone_{arch}_all_s04": "--",
    f"vone_{arch}_mix_s01": "--",
    f"vone_{arch}_mix_s04": "--",
    f"vone_{arch}_random-mix_s00-04": "--",
    f"vone_{arch}_mix_s10": "--",
    f"vone_{arch}_mix_s01_no-blur-1label": "--",
    f"vone_{arch}_mix_s01_no-blur-8label": "--",
    f"vone_{arch}_mix_s04_no-blur-1label": "--",
    f"vone_{arch}_mix_s04_no-blur-8label": "--",
    f"vone_{arch}_mix_s04_no-sharp-1label": "--",
    f"vone_{arch}_mix_s04_no-sharp-8label": "--",
    "resnet50-1x_simclr": "-",
    "resnet50-2x_simclr": "-",
    "resnet50-4x_simclr": "-",
    "humans": "--",
}

    return lines[model_name]