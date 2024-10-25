import random
import re
import os
from tqdm import tqdm


def generate_like_patterns(dataset_file_path, training_data_dir, n_train, n_valid, n_test, max_data_range):
    """
    Generate a set of random LIKE patterns from a given dataset and save them to a file.

    Parameters:
        dataset_file_path (str): The file path of the dataset containing rows to generate patterns from.
        training_data_file (str): The file path to save the generated patterns.
        train_data_size (int): The number of unique LIKE patterns to generate.
    """
    random.seed(0)
    train_data_size = n_train + n_valid + n_test

    with open(dataset_file_path, 'r') as file:
        all_rows = [line.rstrip('\n') for line in file]

    all_patterns = set()
    p_bar = tqdm(total=train_data_size)
    while len(all_patterns) < train_data_size:
        select_random_row = str(random.choice(all_rows))
        len_row = len(select_random_row)

        if max_data_range is not None:
            if len_row > max_data_range:
                new_len = random.randint(
                    max(1, max_data_range//2), max_data_range)
                p_start = random.randint(0, len_row - new_len)
                select_random_row = select_random_row[p_start:p_start+new_len]
                select_random_row = select_random_row.strip()
                len_row = len(select_random_row)
                if len_row == 0:
                    continue
        select_random_row = list(select_random_row)
        n_replace_wild = random.randint(0, len(select_random_row))
        random_indexes = random.sample(
            range(len(select_random_row)), n_replace_wild)
        random_indexes.sort()

        for key in random_indexes:
            if key + 1 not in random_indexes and key - 1 not in random_indexes:
                select_random_row[key] = '_'
            else:
                select_random_row[key] = '%'

        like_pattern = re.sub(r'%+', '%', ''.join(select_random_row))
        if max_data_range is not None:
            n_wild = like_pattern.count('%') + like_pattern.count('_')
            if n_wild > max_data_range:
                continue

        if like_pattern and like_pattern != '%':
            if like_pattern not in all_patterns:
                all_patterns.add(like_pattern)
                p_bar.update(1)

    p_bar.close()
    all_patterns = list(all_patterns)
    os.makedirs(training_data_dir, exist_ok=True)

    pos_dict = {
        'train': [0, n_train],
        'valid': [n_train, n_train + n_valid],
        'test': [n_train + n_valid, -1],
    }

    for q_type in ['train', 'valid', 'test']:
        training_data_file = os.path.join(training_data_dir, f"{q_type}.txt")
        start, end = pos_dict[q_type]

        print(f"Save training queries at {training_data_file = }")
        selected_patterns = all_patterns[start: end]
        if len(selected_patterns) == 0:
            print(f"{q_type = } passed since selected_patterns is empty")
            continue
        with open(training_data_file, 'w') as saving_path:
            saving_path.write(
                '\n'.join(sorted(selected_patterns)) + '\n')
