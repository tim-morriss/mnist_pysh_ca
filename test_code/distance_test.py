from itertools import product


def list_distance(num_list):
    return sum([abs(j - i) for (i, j) in zip(num_list[:-1], num_list[1:])])


if __name__ == '__main__':
    list_1 = [0, 5, 10]
    list_2 = [0, 1, 2]

    print(list_distance(list_1))
    print(list_distance(list_2))
