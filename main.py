import os
import argparse
import re

from mnist_pysh_ca import MNISTPyshCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pop_size', help="PushGP population size (default 50).", nargs='?', default=1, type=int)
    parser.add_argument('-g', '--gens', help="PushGP generations (default 10).", nargs='?', default=1, type=int)
    parser.add_argument('-s', '--steps', help="Steps for the CA (default 25).", nargs='?', default=25, type=int)
    parser.add_argument('-c', '--cut_size', help="Number of samples for each label (default 10).", nargs='?',
                        default=10, type=int)
    parser.add_argument('-d', '--digits', help="Array of digits (default 1,2).", nargs='?', default='1,2', type=str)
    parser.add_argument('-lf', '--load_file', help="File to write to.", nargs='?', default='test-1.txt', type=str)
    parser.add_argument('-sf', '--save_folder', help="Folder to save to.", nargs='?', default='test-1.txt', type=str)
    parser.add_argument('-m', '--mode', help="Training or testing mode.", nargs='?', default='training', type=str)
    parser.add_argument('-r', '--random', help="Use a random sample of dataset.", nargs='?', default='False', type=str)
    parser.add_argument('-simplify', '--simplification', help="Number of simplification steps", nargs='?', default=0,
                        type=int)
    parser.add_argument('-stacks', '--stacks', help="Which stacks to include.", nargs='?', default="float", type=str)
    parser.add_argument('-pic', '--picture', help="Whether or not to output a picture after testing.", nargs='?',
                        default="false", type=str)

    args = parser.parse_args()
    digits = [int(item) for item in args.digits.split(',')]
    shuffle = args.random.lower() == 'true'
    picture = args.picture.lower() == 'true'
    stacks = set([item.strip() for item in re.split(', |,|; | |;', args.stacks)])
    # if not os.path.isdir(args.save_folder):
    #     raise FileNotFoundError("No folder exists with name {0}".format(args.save_folder))

    MNISTPyshCA.mnist_pysh_ca(
        mode=args.mode,
        load_filepath=args.load_file,
        save_folder=args.save_folder,
        pop_size=args.pop_size,
        gens=args.gens,
        steps=args.steps,
        cut_size=args.cut_size,
        digits=digits,
        shuffle=shuffle,
        simplifcation=args.simplification,
        stacks=stacks,
        picture=picture
    )


if __name__ == '__main__':
    main()
