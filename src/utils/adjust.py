def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_multi_steps(epoch):
    """
    for 'multi-steps' mode
    :param blur: flag of whether blur training images or not (default: True)
    :param epoch: training epoch at the moment
    :return: sigma, blur
    """
    """
    if epoch < 10:
        sigma = 5
    elif epoch < 20:
        sigma = 4
    elif epoch < 30:
        sigma = 3
    elif epoch < 40:
        sigma = 2
    elif epoch < 50:
        sigma = 1
    else:
        sigma = 0  # no blur

    """
    if epoch < 10:
        sigma = 4
    elif epoch < 20:
        sigma = 3
    elif epoch < 30:
        sigma = 2
    elif epoch < 40:
        sigma = 1
    else:
        sigma = 0  # no blur

    return sigma


def adjust_multi_steps_cbt(sigma, epoch, decay_rate=0.9, every=5):
    """
    Sets the sigma of Gaussian Blur decayed every 5 epoch.
    This is for 'multi-steps-cbt' mode.
    This idea is based on "Curriculum By Texture"
    :param sigma: blur parameter
    :param blur: flag of whether blur training images or not (default: True)
    :param epoch: training epoch at the moment
    :param decay_rate: how much the model decreases the sigma value
    :param every: the number of epochs the model decrease sigma value
    :return: sigma, blur
    """
    if epoch < every:
        pass
    elif epoch % every == 0:
        sigma = sigma * decay_rate

    # if epoch >= 40:
    #    blur = False

    # return args.init_sigma * (args.cbt_rate ** (epoch // every))
    return sigma
