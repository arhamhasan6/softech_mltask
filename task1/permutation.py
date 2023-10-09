# def generate_permutations(arr):
#     if len(arr) <= 1:
#         return [arr]
    
#     return [[arr[i]] + perm for i in range(len(arr)) for perm in generate_permutations(arr[:i] + arr[i + 1:])]

# user_input = input("Enter a space-separated list of numbers: ")
# # Split the input into a list of integers
# input_data = [int(num) for num in user_input.split()]


# permutations_list = generate_permutations(input_data)

# for perm in permutations_list:
#     print(perm)

import argparse

def generate_permutations(arr):
    if len(arr) <= 1:
        return [arr]

    return [[arr[i]] + perm for i in range(len(arr)) for perm in generate_permutations(arr[:i] + arr[i + 1:])]

def main():
    parser = argparse.ArgumentParser(description="Generate permutations of a list of numbers.")
    parser.add_argument("numbers", type=int, nargs="+", help="List of numbers separated by spaces")
    args = parser.parse_args()
    
    permutations_list = generate_permutations(args.numbers)

    for perm in permutations_list:
        print(perm)

if __name__ == "__main__":
    main()
