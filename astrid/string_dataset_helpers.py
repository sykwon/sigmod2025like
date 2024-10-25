import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import re
import src.util as ut
from tqdm import tqdm


class StringDatasetHelper:
    def __init__(self):
        self.alphabets = ""
        self.alphabet_size = 0
        self.max_string_length = 0

    def set_max_string_length(self, max_string_length):
        self.max_string_length = max_string_length
        if self.max_string_length < 8:
            self.max_string_length = 8

    def extract_alphabets(self, file_name):
        f = open(file_name, "r")
        lines = f.read().splitlines()
        f.close()

        self.max_string_length = max(map(len, lines))
        # Make it to even for easier post processing and striding
        if self.max_string_length % 2 != 0:
            self.max_string_length += 1

        # Get the alphabets. Create a set with all strings, sort it and concatenate it
        wild_cards = "_%"
        MAX_CHAR_SIZE = 200
        alphabets_list = sorted(set().union(*lines))
        for wild_card in wild_cards:
            if wild_card in alphabets_list:
                alphabets_list.remove(wild_card)

        if len(alphabets_list) > MAX_CHAR_SIZE:
            char_count = Counter("".join(lines))
            alphabets_list = sorted(
                sorted(char_count.keys()), key=lambda x: char_count[x], reverse=True)
            for wild_card in wild_cards:
                if wild_card in alphabets_list:
                    alphabets_list.remove(wild_card)
            alphabets_list = alphabets_list[:MAX_CHAR_SIZE]
        self.alphabets = "".join(alphabets_list)
        assert "_" not in self.alphabets
        assert "%" not in self.alphabets

        self.alphabets = "팯언%_" + self.alphabets
        self.alphabet_size = len(self.alphabets)
        # print(self.alphabets)
        # print(f"{self.alphabet_size = }")

    # If alphabet is "abc" and for the word abba, this returns [0, 1, 1, 0]
    # 0 is the index for a and 1 is the index for b
    def string_to_ids(self, str_val):
        # if len(str_val) > self.max_string_length:
        #     print("Warning: long string {} is passed. Subsetting to max length of {}".format(str_val, self.max_string_length))
        #     str_val = str[:self.max_string_length]
        assert len(str_val) > 0
        tokens = str_val.strip("%").split("%")
        if tokens[0] == '':
            tokens = ['%']

        indices = []
        for token in tokens:
            # 1 means unknown
            indices.append([self.alphabets.find(
                c) if c in token else 1 for c in token])
        if len(indices) == 1:
            indices = indices[0]
        # if -1 in indices:
        #     raise ValueError("String {} contained unknown alphabets".format(str_val))
        return indices

    # Given a string (of any length), it outputs a fixed 2D tensor of size alphabet_size * max_string_length
    # If the string is shorter, the rest are filled with zeros
    # Each column corresponds to the i-th character of str_val
    # while each row corresponds to j-th character of self.alphabets
    # This encoding is good for CNN processing
    def string_to_tensor(self, str_val):
        string_indices = self.string_to_ids(str_val)
        if isinstance(string_indices[0], list):
            one_hot_tensor = np.zeros((len(string_indices), self.alphabet_size,
                                      self.max_string_length), dtype=np.float32)
            for i in range(len(string_indices)):
                one_hot_tensor[i, np.array(string_indices[i]), np.arange(
                    len(string_indices[i]))] = 1.0
        else:
            one_hot_tensor = np.zeros(
                (self.alphabet_size, self.max_string_length), dtype=np.float32)
            one_hot_tensor[np.array(string_indices), np.arange(
                len(string_indices))] = 1.0
        return torch.from_numpy(one_hot_tensor)


# Helper string to process file with three strings in each line
# It also automatically converts each string to a tensor
class TripletStringDataset(Dataset):
    def __init__(self, df, string_helper):
        # Convert data frame to numpy ndarray
        self.data = df.values
        self.string_helper: StringDatasetHelper = string_helper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Get the three strings, convert to tensor and return it
        # print("Item, data", item, self.data[item])
        # str(val) is a precaution for special cases such as strings like 'nan' which confuses pandas
        anchor, positive, negative = [self.string_helper.string_to_tensor(
            str(val)) for val in self.data[item]]
        return anchor, positive, negative

# Helper string to process file strings and automatically convert to embeddings


def sql_LIKE_query_to_embedding_vector(query_str, string_helper: StringDatasetHelper, embedding_model_dict):
    query_str = ut.canonicalize_like_query(query_str, True)
    # idx = ut.find_like_query_type_idx(query_str)
    tokens = list(filter(''.__ne__, query_str.split("%")))
    fn_desc_list = []

    fn_desc_list = ["substring" for _ in range(len(tokens))]
    if query_str[0] == "%" and query_str[-1] == "%":
        pass
    elif query_str[-1] == "%":
        fn_desc_list[0] = "prefix"
    elif query_str[0] == "%":
        fn_desc_list[-1] = "suffix"

    n_pat = len(tokens)
    embeds_list = []
    for fn_desc, token in zip(fn_desc_list, tokens):
        embedding_model = embedding_model_dict[fn_desc]
        token_as_tensor = string_helper.string_to_tensor(token)
        token_as_tensor = token_as_tensor.reshape(1, *token_as_tensor.shape)
        embeds = embedding_model(token_as_tensor).numpy()
        embeds_list.append(embeds)

    output = np.vstack(embeds_list)
    return output, n_pat


class StringSelectivityDataset(Dataset):
    def __init__(self, df, string_helper, embedding_model_dict):
        self.strings = df["string"].values.tolist()
        self.normalized_selectivities = df["normalized_selectivities"].values.tolist(
        )
        self.strings_as_tensors = []
        self.n_pat_list = []
        self.string_helper = string_helper
        self.embedding_model_dict = embedding_model_dict
        max_n_pat = 2

        with torch.no_grad():
            for idx, string in enumerate(self.strings, start=1):
                # string_as_tensor = string_helper.string_to_tensor(string)
                # # By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
                # # so create a "fake" dimension that converts the 2D matrix into a 3D tensor
                # if string_as_tensor.dim() == 2:
                #     string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
                # embeds = self.embedding_model(string_as_tensor).numpy()
                # n_pat = len(embeds)
                embeds, n_pat = sql_LIKE_query_to_embedding_vector(
                    string, string_helper, embedding_model_dict)
                self.n_pat_list.append(n_pat)

                if n_pat < max_n_pat:
                    embeds = np.pad(
                        embeds, ((0, max_n_pat-n_pat), (0, 0)), 'constant', constant_values=0)
                embeds = embeds.reshape(1, *embeds.shape)
                self.strings_as_tensors.append(embeds)
            self.strings_as_tensors = np.concatenate(self.strings_as_tensors)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, item):
        return self.strings_as_tensors[item], self.n_pat_list[item], self.normalized_selectivities[item]


class StringSelectivityDatasetAstridEach(Dataset):
    def __init__(self, df, string_helper, embedding_model_dict, is_tqdm=False):
        self.strings = df["string"].values
        self.normalized_selectivities = df["normalized_selectivities"].values
        self.strings_as_tensors = []
        self.string_helper = string_helper
        self.embedding_model_dict = embedding_model_dict

        for fn_desc, embedding_model in self.embedding_model_dict.items():
            embedding_model.eval()

        if is_tqdm:
            strings = tqdm(self.strings, desc="Dataset")
        else:
            strings = self.strings
        with torch.no_grad():
            for string in strings:
                if string[0] == "%" and string[-1] == "%":
                    fn_desc == "substring"
                elif string[0] != "%":
                    fn_desc == "prefix"
                elif string[-1] != "%":
                    fn_desc == "suffix"

                embedding_model = embedding_model_dict[fn_desc]
                string_as_tensor = string_helper.string_to_tensor(string)
                # By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
                # so create a "fake" dimension that converts the 2D matrix into a 3D tensor
                string_as_tensor = string_as_tensor.view(
                    -1, *string_as_tensor.shape)
                self.strings_as_tensors.append(
                    embedding_model(string_as_tensor).numpy())
            self.strings_as_tensors = np.concatenate(self.strings_as_tensors)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, item):
        return self.strings_as_tensors[item], self.normalized_selectivities[item]
