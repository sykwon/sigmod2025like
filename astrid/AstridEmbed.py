import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pandas as pd
import astrid.misc_utils as misc_utils
from astrid.string_dataset_helpers import TripletStringDataset, StringSelectivityDataset, StringSelectivityDatasetAstridEach
import astrid.EmbeddingLearner as EmbeddingLearner
import astrid.SupervisedSelectivityEstimator as SupervisedSelectivityEstimator
from astrid.misc_utils import initialize_random_seeds, setup_vocabulary
import os
import time
import pickle
import yaml
from astrid.EmbeddingLearner import EmbeddingCNNNetwork
# embedding_learner_configs, frequency_configs, selectivity_learner_configs = None, None, None


# This function gives a single place to change all the necessary configurations.
# Please see misc_utils for some additional descriptions of what these attributes mean
def setup_configs(data_name, bs, lr, seed, pack_type, max_n_pat, tag, min_val, max_val, emb_dim=64, emb_epoch=32, est_epoch=64, query_type="substring", agg=None, **kwargs):
    # global embedding_learner_configs, frequency_configs, selectivity_learner_configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_learner_configs = misc_utils.AstridEmbedLearnerConfigs(embedding_dimension=emb_dim, batch_size=bs,
                                                                     num_epochs=emb_epoch, margin=0.2, device=device, lr=lr, channel_size=8)

    # path = "datasets/dblp/"
    input_path = f"data/{data_name}/"
    output_path = f"res/{data_name}/{seed}/model/"

    # This assumes that prepare_dataset function was called to output the files.
    # If not, please change the file names appropriately
    # file_name_prefix = "dblp_titles"
    # query_type = "prefix"
    frequency_configs = misc_utils.StringFrequencyConfigs(
        string_list_file_name=f"data/{data_name}/{data_name}.txt",
        selectivity_file_name=input_path +
        f"training/{pack_type}_{max_n_pat}.txt",
        triplets_file_name=input_path + f"triplets/{query_type}.txt"
    )

    selectivity_learner_configs = misc_utils.SelectivityEstimatorConfigs(
        embedding_dimension=emb_dim, batch_size=bs, num_epochs=est_epoch, device=device, lr=lr,
        # will be updated in train_selectivity_estimator
        min_val=min_val, max_val=max_val,
        # embedding_model_file_name=output_path + f"{tag}{pack_type}_{max_n_pat}_embed.pth",
        agg=agg,
        embedding_model_file_name=input_path + "emb_model/" + f"embed.pth",
        selectivity_model_file_name=output_path + \
        f"{tag}{pack_type}_{max_n_pat}.pth",
    )

    return embedding_learner_configs, frequency_configs, selectivity_learner_configs


# This function trains and returns the embedding model
def train_astrid_embedding_model(string_helper, embedding_learner_configs: misc_utils.AstridEmbedLearnerConfigs, triplets_file_name, model_output_file_name=None, prepare_time=None):
    initialize_random_seeds(0)
    # global embedding_learner_configs, frequency_configs

    is_train = triplets_file_name is not None
    MODE_SIZE = embedding_learner_configs.num_epochs == 0
    if is_train:
        assert model_output_file_name is not None
        assert prepare_time is not None
        print(f"{model_output_file_name = }")
    build_time_output_file_name = model_output_file_name.replace(
        '.pth', '.yml')
    print(f"{build_time_output_file_name = }")

    if is_train and not (os.path.exists(model_output_file_name) and os.path.exists(build_time_output_file_name)):

        start_time = time.time()
        if MODE_SIZE:
            train_loader = None
        else:
            batch_size = embedding_learner_configs.batch_size

            # Some times special strings such as nan or those that start with a number confuses Pandas
            df = pd.read_csv(triplets_file_name)
            df["Anchor"] = df["Anchor"].astype(str)
            df["Positive"] = df["Positive"].astype(str)
            df["Negative"] = df["Negative"].astype(str)

            triplet_dataset = TripletStringDataset(df, string_helper)
            train_loader = DataLoader(
                triplet_dataset, batch_size, shuffle=True)

        # if not os.path.exists(model_output_file_name):
        embedding_model = EmbeddingLearner.train_embedding_model(
            embedding_learner_configs, train_loader, string_helper)

        end_time = time.time()

        learn_emb_time = (end_time - start_time)
        build_time = prepare_time + learn_emb_time

        assert model_output_file_name is not None
        os.makedirs(os.path.dirname(model_output_file_name), exist_ok=True)
        torch.save(embedding_model.state_dict(), model_output_file_name)

        outdict = {'prepare_time': prepare_time, 'learn_emb_time': learn_emb_time,
                   'build_time': build_time}

        with open(build_time_output_file_name, 'w') as f:
            yaml.safe_dump(outdict, f)

    embedding_model = EmbeddingCNNNetwork(
        string_helper, embedding_learner_configs)
    embedding_model.load_state_dict(torch.load(model_output_file_name))
    embedding_model = embedding_model.to(embedding_learner_configs.device)

    with open(build_time_output_file_name) as f:
        outdict = yaml.safe_load(f)
        build_time = outdict['build_time']
    return embedding_model, build_time

# This function performs min-max scaling over logarithmic data.
# Typically, the selectivities are very skewed.
# This transformation reduces the skew and makes it easier for DL to learn the models


def compute_normalized_selectivities(df, min_val, max_val):
    # global selectivity_learner_configs
    # normalized_selectivities, min_val, max_val = misc_utils.normalize_labels(df["selectivity"])
    normalized_selectivities, min_val, max_val = misc_utils.normalize_labels(
        df["selectivity"], min_val, max_val)
    df["normalized_selectivities"] = normalized_selectivities

    # namedtuple's are immutable - so replace them with new instances
    # selectivity_learner_configs = selectivity_learner_configs._replace(min_val=min_val)
    # selectivity_learner_configs = selectivity_learner_configs._replace(max_val=max_val)
    return df


# This function trains and returns the selectivity estimator.
def train_selectivity_estimator(train_loader, valid_loader, test_loader, string_helper, selectivity_learner_configs, est_scale, patience, summary_writer=None):
    # def train_selectivity_estimator(train_df, string_helper, embedding_model, model_output_file_name=None):
    # global selectivity_learner_configs, frequency_configs

    # string_dataset = StringSelectivityDataset(train_df, string_helper, embedding_model)
    # train_loader = DataLoader(string_dataset, batch_size=selectivity_learner_configs.batch_size, shuffle=True)

    selectivity_model = SupervisedSelectivityEstimator.train_selEst_model(
        selectivity_learner_configs, train_loader, valid_loader, test_loader, string_helper, est_scale, patience, sw=summary_writer)
    # if model_output_file_name is not None:
    #     torch.save(selectivity_model.state_dict(), model_output_file_name)
    return selectivity_model


def train_selectivity_estimator_AstridEach(train_loader, valid_loader, test_loader, string_helper, selectivity_learner_configs, patience, summary_writer=None, is_train=True):
    selectivity_model = SupervisedSelectivityEstimator.train_selEst_model_AstridEach(
        selectivity_learner_configs, train_loader, valid_loader, test_loader, string_helper, patience, sw=summary_writer, is_train=is_train)
    return selectivity_model

# This is a helper function to get selectivity estimates for an iterator of strings


def get_selectivity_for_strings(strings, embedding_model_dict, selectivity_model, string_helper, selectivity_learner_configs):
    # global selectivity_learner_configs
    # from astrid.SupervisedSelectivityEstimator import SelectivityEstimator
    # embedding_model_dict.eval()
    # selectivity_model.eval()
    strings_as_tensors = []
    min_val = selectivity_learner_configs.min_val
    max_val = selectivity_learner_configs.max_val
    df_test = data2Astrid_df([(x, 1) for x in strings], min_val, max_val)
    ds_test = StringSelectivityDataset(
        df_test, string_helper, embedding_model_dict)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    configs = selectivity_learner_configs
    with torch.no_grad():
        normalized_predictions = []
        for string_queries, n_pats, true_selectivities in dl_test:
            # for string in strings:
            # string_as_tensor = string_helper.string_to_tensor(string)
            # By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
            # so create a "fake" dimension that converts the 2D matrix into a 3D tensor
            # if configs.agg == "pool":
            #     string_queries = torch.sum(string_queries, dim=1) / n_pats.view(-1, 1)
            # elif configs.agg == "attn":
            #     pass
            # string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
            # strings_as_tensors.append(embedding_model(string_as_tensor).numpy())
            normalized_predictions.append(selectivity_model(
                (string_queries, n_pats), agg=configs.agg).view(-1))
        # strings_as_tensors = np.concatenate(strings_as_tensors)
        # normalized_selectivities= between 0 to 1 after the min-max and log scaling.
        # denormalized_predictions are the frequencies between 0 to N
        # normalized_predictions = selectivity_model(torch.tensor(strings_as_tensors))
        normalized_predictions = torch.cat(normalized_predictions)
        denormalized_predictions = misc_utils.unnormalize_torch(
            normalized_predictions, selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
        return normalized_predictions, denormalized_predictions


def get_selectivity_for_strings_AstridEach(strings, embedding_model_dict, selectivity_model_dict, string_helper, selectivity_learner_configs):
    # global selectivity_learner_configs
    # from astrid.SupervisedSelectivityEstimator import SelectivityEstimator
    # embedding_model_dict.eval()
    # selectivity_model.eval()
    strings_as_tensors = []
    min_val = selectivity_learner_configs.min_val
    max_val = selectivity_learner_configs.max_val
    df_test = data2Astrid_df([(x, 1) for x in strings], min_val, max_val)
    ds_test = StringSelectivityDatasetAstridEach(
        df_test, string_helper, embedding_model_dict)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    configs = selectivity_learner_configs
    with torch.no_grad():
        normalized_predictions = []
        for string, (string_queries, true_selectivities) in zip(strings, dl_test):
            if "%" == string[0] and "%" == string[-1]:
                fn_desc = 'substring'
            elif "%" != string[0]:
                fn_desc = 'prefix'
            elif "%" != string[-1]:
                fn_desc = 'suffix'
            selectivity_model = selectivity_model_dict[fn_desc]
            normalized_predictions.append(
                selectivity_model(string_queries).view(-1))
        # strings_as_tensors = np.concatenate(strings_as_tensors)
        # normalized_selectivities= between 0 to 1 after the min-max and log scaling.
        # denormalized_predictions are the frequencies between 0 to N
        # normalized_predictions = selectivity_model(torch.tensor(strings_as_tensors))
        normalized_predictions = torch.cat(normalized_predictions)
        denormalized_predictions = misc_utils.unnormalize_torch(
            normalized_predictions, selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
        return normalized_predictions, denormalized_predictions


def load_embedding_model(model_file_name, string_helper, embedding_learner_configs):
    from astrid.EmbeddingLearner import EmbeddingCNNNetwork
    embedding_model = EmbeddingCNNNetwork(
        string_helper, embedding_learner_configs)
    embedding_model.load_state_dict(torch.load(model_file_name))
    return embedding_model


def load_selectivity_estimation_model(model_file_name, string_helper, selectivity_learner_configs, est_scale):
    from SupervisedSelectivityEstimator import SelectivityEstimator
    selectivity_model = SelectivityEstimator(
        string_helper, selectivity_learner_configs, est_scale)
    selectivity_model.load_state_dict(torch.load(model_file_name))
    return selectivity_model


def data2Astrid_df(data, min_val, max_val):
    header = ["string", "selectivity"]
    df = pd.DataFrame(data, columns=header)
    df = compute_normalized_selectivities(df, min_val, max_val)
    return df

# def main():
#     random_seed = 1234
#     misc_utils.initialize_random_seeds(random_seed)

#     # Set the configs
#     embedding_learner_configs, frequency_configs, selectivity_learner_configs = setup_configs()

#     embedding_model_file_name = selectivity_learner_configs.embedding_model_file_name
#     selectivity_model_file_name = selectivity_learner_configs.selectivity_model_file_name

#     string_helper = misc_utils.setup_vocabulary(frequency_configs.string_list_file_name)

#     # You can comment/uncomment the following lines based on whether you
#     # want to train from scratch or just reload a previously trained embedding model.
#     embedding_model = train_astrid_embedding_model(string_helper, embedding_model_file_name)
#     #embedding_model = load_embedding_model(embedding_model_file_name, string_helper)

#     # Load the input file and split into 50-50 train, test split
#     df = pd.read_csv(frequency_configs.selectivity_file_name)
#     # Some times strings that start with numbers or
#     # special strings such as nan which confuses Pandas' type inference algorithm
#     df["string"] = df["string"].astype(str)
#     df = compute_normalized_selectivities(df)
#     train_indices, test_indices = train_test_split(df.index, random_state=random_seed, test_size=0.5)
#     train_df, test_df = df.iloc[train_indices], df.iloc[test_indices]

#     # You can comment/uncomment the following lines based on whether you
#     # want to train from scratch or just reload a previously trained embedding model.
#     selectivity_model = train_selectivity_estimator(train_df, string_helper,
#                                                     embedding_model, selectivity_model_file_name)
#     #selectivity_model = load_selectivity_estimation_model(selectivity_model_file_name, string_helper)

#     # Get the predictions from the learned model and compute basic summary statistics
#     normalized_predictions, denormalized_predictions = get_selectivity_for_strings(
#         test_df["string"].values, embedding_model, selectivity_model, string_helper)
#     actual = torch.tensor(test_df["normalized_selectivities"].values)
#     test_q_error = misc_utils.compute_qerrors(normalized_predictions, actual,
#                                               selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
#     print("Test data: Mean q-error loss ", np.mean(test_q_error))
#     print("Test data: Summary stats of Loss: Percentile: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] ", [
#           np.quantile(test_q_error, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]])


# if __name__ == "__main__":
#     main()
