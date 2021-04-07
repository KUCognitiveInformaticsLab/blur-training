def print_settings(model, args):
    print("=" * 5 + " settings " + "=" * 5)
    print("TRAINING MODE: {}".format(args.mode))
    if args.mode == "mix":
        print("### MIX ###")
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "single-step":
        print("### SINGLE STEP ###")
        print("## NO BLUR FROM EPOCH {:d}".format(args.epochs // 2))
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "reversed-single-step":
        print("### REVERSED SINGLE STEP ###")
        print("## NO BLUR TILL EPOCH {:d}".format(args.epochs // 2))
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "multi-steps":
        print("### MULTI STEPS ###")
        print(
            "Step: 1-10 -> 11-20 -> 21-30 -> 31-40 -> 41-50 -> 51-{}".format(
                args.epochs
            )
        )
        print("Sigma: 5 -> 4 -> 3 -> 2 -> 1 -> none")
        print("#" * 20)
    elif args.mode == "all":
        print("Sigma: {}".format(args.sigma))
    if args.blur_val:
        print("VALIDATION MODE: blur-val")
    print("Random seed: {}".format(args.seed))
    print("Epochs: {}".format(args.epochs))
    print("Learning rate: {}".format(args.lr))
    print("Weight_decay: {}".format(args.weight_decay))
    print()
    print(model)
    print("=" * 20)
    print()
