import argparse
import statistics
import csv

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('list', help="list of errors", nargs='?', default="", type=str)
    parse.add_argument('digits', help="which digits in what order", nargs='?', default="1,2", type=str)
    parse.add_argument('-n', '--number', help="Number of samples for each label (default 10).", nargs='?', default=10, type=int)
    parse.add_argument('-f', '--filename', help="file to load error vector from", nargs='?', default=None, type=str)

    args = parse.parse_args()
    digits = [int(item) for item in args.digits.split(',')]

    if args.filename is None:
        errors = [float(item) for item in args.list.split()]
    else:
        with open(args.filename, 'r') as f:
            errors = []
            reader = csv.reader(f, delimiter=' ')
            for item in reader:
                for number in item:
                    errors.append(float(number))
        f.close()

    error_digit_1 = sum(errors[:args.number])
    error_digit_2 = sum(errors[args.number:])
    print("Total error for {0}: \n {1}".format(digits[0], error_digit_1))
    print("Total error for {0}: \n {1}".format(digits[1], error_digit_2))

    avg_digit_1 = statistics.mean(errors[:args.number])
    avg_digit_2 = statistics.mean(errors[args.number:])
    print("Average error for {0}: \n {1}".format(digits[0], avg_digit_1))
    print("Average error for {0}: \n {1}".format(digits[1], avg_digit_2))

    predicted_1 = [(digits[0] - num) for num in errors[:args.number]]
    predicted_2 = [(digits[1] - num) for num in errors[args.number:]]
    print("Predicted values for {0}: \n {1}".format(digits[0], predicted_1))
    print("Predicted values for {0}: \n {1}".format(digits[1], predicted_2))
