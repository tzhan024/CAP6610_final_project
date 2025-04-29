import argparse
import json


def is_computable(x):
    """Return True if x is a numeric value, a dict, or a list of computable items."""
    if isinstance(x, (int, float)):
        return True
    elif isinstance(x, dict):
        return True
    elif isinstance(x, list):
        # For lists, if empty, we'll treat it as computable.
        if not x:
            return True
        # Ensure all items are of the same type and computable.
        first_type = type(x[0])
        for item in x:
            if type(item) != first_type or not is_computable(item):
                return False
        return True
    return False  # Anything else (like str) is not computable

def calculate_average(dict_list):
    if not dict_list:
        return {}
    
    # Assume all dictionaries have the same keys.
    keys = dict_list[0].keys()
    result = {}
    
    for key in keys:
        # Skip the key if any of the values in the list are not computable.
        if not all(is_computable(d[key]) for d in dict_list):
            continue
        
        first_value = dict_list[0][key]
        
        if isinstance(first_value, dict):
            # All d[key] are dicts.
            nested_list = [d[key] for d in dict_list]
            result[key] = calculate_average(nested_list)
        
        elif isinstance(first_value, list):
            # Ensure all d[key] are lists with the same length.
            if not all(isinstance(d[key], list) for d in dict_list):
                continue
            list_length = len(first_value)
            if not all(len(d[key]) == list_length for d in dict_list):
                continue

            # If the list is empty, return an empty list.
            if list_length == 0:
                result[key] = []
                continue

            # Process lists element-wise.
            # Check the type of the first element to decide whether to recurse or average numerically.
            first_elem = first_value[0]
            averaged_list = []
            if isinstance(first_elem, dict):
                for i in range(list_length):
                    # Gather the i-th element from each dictionary's list.
                    elements = [d[key][i] for d in dict_list]
                    # If any element is not a dict, skip the key.
                    if not all(isinstance(el, dict) for el in elements):
                        averaged_list = None
                        break
                    averaged_list.append(calculate_average(elements))
            elif isinstance(first_elem, (int, float)):
                for i in range(list_length):
                    elements = [d[key][i] for d in dict_list]
                    if not all(isinstance(el, (int, float)) for el in elements):
                        averaged_list = None
                        break
                    averaged_list.append(sum(elements) / len(elements))
            else:
                # Unsupported type inside list, skip key.
                averaged_list = None
            
            if averaged_list is not None:
                result[key] = averaged_list
        
        elif isinstance(first_value, (int, float)):
            # Compute the average for numeric values.
            if all(isinstance(d[key], (int, float)) for d in dict_list):
                result[key] = sum(d[key] for d in dict_list) / len(dict_list)
        
        # If the value is not numeric, dict, or list (or not consistent), skip the key.
    
    return result

# Example usage:
# data = [
#     {
#         'a': 1,
#         'b': {'x': 10, 'y': 20},
#         'c': [1, 2, 3],
#         'd': "non-computable"  # This key will be skipped.
#     },
#     {
#         'a': 3,
#         'b': {'x': 30, 'y': 40},
#         'c': [4, 5, 6],
#         'd': "skip me"
#     },
#     {
#         'a': 5,
#         'b': {'x': 50, 'y': 60},
#         'c': [7, 8, 9],
#         'd': "still skip"
#     }
# ]

# averages = calculate_average(data)
# print(averages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="file path",
        type=str,
        required = False
    )
    
    args = parser.parse_args()
    filename = args.file
    print(f"FILENAME: <{filename}>")
    with open(filename, 'r') as f:
        data = json.load(f)

    averages = calculate_average(data)
    dump_json = json.dumps(averages, indent=4)
    print(dump_json)
    # print(averages['sc_token_count'])
    print(f"sc actual compress rate: {float(averages['sc_token_count']) / float(averages['context_token_count'])}")
    print(f"lingua2 actual compress rate: {float(averages['lingua2_token_count']) / float(averages['context_token_count'])}")
    print('\n\n')
