import csv


def save_acc1(acc1, save_file):
    with open(save_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["acc1"])
        writer.writerow([acc1])
