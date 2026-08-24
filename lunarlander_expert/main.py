import argparse
from train.train import train
from test.test import test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
                        type=str, 
                        required=True, 
                        choices=["train", "test"])
    args = parser.parse_args()

    if args.mode == "train":
        print("START TRAIN")   # DEBUG
        train()

    elif args.mode == "test":
        print("START TEST")
        test()


if __name__ == "__main__":
    main()