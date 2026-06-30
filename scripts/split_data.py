import argparse
import os
import random
import shutil

# Keep split utility independent from torch imports.
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    return parser.parse_args()


def copy_files(file_list, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for file_name in file_list:
        shutil.copy2(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))


def main():
    args = parse_args()
    assert abs((args.train + args.val + args.test) - 1.0) < 1e-6, "Ratios must sum to 1."

    random.seed(RANDOM_SEED)
    class_dirs = [d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))]

    for cls in class_dirs:
        cls_src = os.path.join(args.source, cls)
        images = [f for f in os.listdir(cls_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * args.train)
        n_val = int(n_total * args.val)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        copy_files(train_files, cls_src, os.path.join(args.dest, "train", cls))
        copy_files(val_files, cls_src, os.path.join(args.dest, "val", cls))
        copy_files(test_files, cls_src, os.path.join(args.dest, "test", cls))

        print(f"{cls}: total={n_total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
