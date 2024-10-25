from src.estimator import Estimator
from astrid.prepare_datasets import prepare_dataset_each_type
from astrid.SupervisedSelectivityEstimator import SelectivityEstimatorAstridEach
from astrid.AstridEmbed import (
    setup_configs,
    train_astrid_embedding_model,
    data2Astrid_df,
    train_selectivity_estimator_AstridEach,
    StringSelectivityDatasetAstridEach,
    get_selectivity_for_strings_AstridEach,
)
from astrid.misc_utils import (
    initialize_random_seeds,
    setup_vocabulary,
    StringDatasetHelper,
)
import astrid.misc_utils
import torch.nn as nn
import torch
import os
import time
from torch.utils.data import DataLoader
import pandas as pd
import src.util as ut


class AstridEachEstimator(nn.Module, Estimator):
    def __init__(
        self,
        data_name,
        emb_dim,
        emb_bs,
        emb_lr,
        emb_epoch,
        db_path,
        triplet_dirpath,
        embedding_model_dirpath,
        model_path,
        bs,
        lr,
        patience,
        est_epoch,
        tag,
        seed,
        min_val,
        max_val,
        max_str_size,
    ):
        super().__init__()
        self.data_name = data_name
        self.emb_dim = emb_dim
        self.emb_bs = emb_bs
        self.emb_lr = emb_lr
        self.emb_epoch = emb_epoch
        self.db_path = db_path
        self.triplet_dirpath = triplet_dirpath
        self.embedding_model_dirpath = embedding_model_dirpath
        self.model_path = model_path
        self.bs = bs
        self.lr = lr
        self.patience = patience
        self.est_epoch = est_epoch
        self.tag = tag
        self.seed = seed
        self.min_val = min_val
        self.max_val = max_val
        self.max_str_size = max_str_size
        self.fn_desc_list = ["prefix", "suffix", "substring"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.string_helper = self.setup_string_helper(db_path)
        self.selectivity_learner_configs_triple = self.setup_selectivity_learner_configs_triple()
        self.embedding_model_dict = {}
        self.selectivity_model_dict = {}

    def setup_string_helper(self, db_path):
        max_str_size = self.max_str_size
        string_helper: StringDatasetHelper = setup_vocabulary(db_path)
        string_helper.set_max_string_length(max_str_size)
        print(f"{string_helper.alphabet_size = }")
        print(f"{string_helper.alphabets = }")
        print(f"{string_helper.max_string_length = }")
        self.string_helper = string_helper
        return string_helper

    def setup_selectivity_learner_configs_triple(self):
        emb_dim = self.emb_dim
        bs = self.bs
        est_epoch = self.est_epoch
        device = self.device
        lr = self.lr
        min_val = self.min_val
        max_val = self.max_val
        model_path = self.model_path
        fn_desc_list = self.fn_desc_list
        selectivity_learner_configs_triple = []
        for fn_desc in fn_desc_list:
            model_path_each = model_path.replace(
                "AstridEach", f"AstridEach/{fn_desc}")
            os.makedirs(os.path.dirname(model_path_each), exist_ok=True)
            selectivity_learner_configs = astrid.misc_utils.SelectivityEstimatorEachConfigs(
                embedding_dimension=emb_dim,
                batch_size=bs,
                num_epochs=est_epoch,
                device=device,
                lr=lr,
                min_val=min_val,
                max_val=max_val,
                embedding_model_file_name="",  # 'input_path + "emb_model/" + f"embed.pth",
                selectivity_model_file_name=model_path_each,
            )
            selectivity_learner_configs_triple.append(
                selectivity_learner_configs)
        return selectivity_learner_configs_triple

    def get_embedding_learner_configs(self):
        device = self.device
        emb_dim = self.emb_dim
        emb_bs = self.emb_bs
        emb_epoch = self.emb_epoch
        emb_lr = self.emb_lr
        embedding_learner_configs = astrid.misc_utils.AstridEmbedLearnerConfigs(
            embedding_dimension=emb_dim,
            batch_size=emb_bs,
            num_epochs=emb_epoch,
            margin=0.2,
            device=device,
            lr=emb_lr,
            channel_size=8,
        )
        self.embedding_learner_configs = embedding_learner_configs
        return embedding_learner_configs

    def build_embedding_models(self):
        device = self.device
        max_str_size = self.max_str_size
        fn_desc_list = self.fn_desc_list
        triplet_dirpath = self.triplet_dirpath
        embedding_model_dirpath = self.embedding_model_dirpath
        db_path = self.db_path
        string_helper = self.string_helper

        embedding_learner_configs = self.get_embedding_learner_configs()

        embedding_model_dict = self.embedding_model_dict
        total_build_time = 0
        for fn_desc in fn_desc_list:
            start_time = time.time()
            triplets_file_name = os.path.join(
                triplet_dirpath, f"{fn_desc}_triplets.csv"
            )
            if not os.path.exists(triplets_file_name):
                triplets_file_name = prepare_dataset_each_type(
                    db_path, triplet_dirpath, max_str_size, fn_desc
                )
            end_time = time.time()
            prepare_time = end_time - start_time
            print(f"{fn_desc = } {triplets_file_name = }")
            embedding_model_file_name = os.path.join(
                embedding_model_dirpath, f"{fn_desc}.pth"
            )
            print(f"{fn_desc = } {embedding_model_file_name = }")
            if not os.path.exists(embedding_model_file_name):
                embedding_model, build_time = train_astrid_embedding_model(
                    string_helper,
                    embedding_learner_configs,
                    triplets_file_name,
                    embedding_model_file_name,
                    prepare_time=prepare_time,
                )
            else:
                embedding_model, build_time = train_astrid_embedding_model(
                    string_helper,
                    embedding_learner_configs,
                    None,
                    embedding_model_file_name,
                    prepare_time=prepare_time,
                )
            embedding_model_dict[fn_desc] = embedding_model.cpu().eval()
            print(f"{fn_desc = }, {build_time = }")
            total_build_time += build_time

        return embedding_model_dict, total_build_time

    def build_model(self, train_data_triple, valid_data_triple, test_data_triple, embedding_model_dict, sw, is_train=True):
        min_val = self.min_val
        max_val = self.max_val
        string_helper = self.string_helper
        bs = self.bs
        seed = self.seed
        device = self.device
        lr = self.lr
        est_epoch = self.est_epoch
        emb_dim = self.emb_dim
        model_path = self.model_path
        patience = self.patience
        selectivity_learner_configs_triple = self.selectivity_learner_configs_triple
        fn_desc_list = self.fn_desc_list
        MODE_SIZE = est_epoch == 0

        for idx, fn_desc in enumerate(fn_desc_list):
            selectivity_learner_configs = selectivity_learner_configs_triple[idx]
            if MODE_SIZE:
                dl_train = None
                dl_valid = None
                dl_test = None
            elif is_train:
                train_data = train_data_triple[idx]
                valid_data = valid_data_triple[idx]
                test_data = test_data_triple[idx]
                embedding_model = embedding_model_dict[fn_desc]

                df_train = data2Astrid_df(train_data, min_val, max_val)
                df_valid = data2Astrid_df(valid_data, min_val, max_val)
                df_test = data2Astrid_df(test_data, min_val, max_val)

                ds_train = StringSelectivityDatasetAstridEach(
                    df_train, string_helper, embedding_model_dict, is_tqdm=True
                )
                ds_valid = StringSelectivityDatasetAstridEach(
                    df_valid, string_helper, embedding_model_dict, is_tqdm=True
                )
                ds_test = StringSelectivityDatasetAstridEach(
                    df_test, string_helper, embedding_model_dict, is_tqdm=True
                )

                dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
                dl_valid = DataLoader(ds_valid, batch_size=bs, shuffle=False)
                dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False)

                initialize_random_seeds(seed)

            selectivity_model: SelectivityEstimatorAstridEach = train_selectivity_estimator_AstridEach(
                dl_train,
                dl_valid,
                dl_test,
                string_helper,
                selectivity_learner_configs,
                patience,
                summary_writer=sw,
                is_train=is_train,
            )

            self.selectivity_model_dict[fn_desc] = selectivity_model

        return self.selectivity_model_dict

    def build(self, train_data_triple, valid_data_triple, test_data_triple, sw, seed):
        initialize_random_seeds(seed)
        embedding_model_dict, build_time = self.build_embedding_models()
        start_time = time.time()
        self.build_model(train_data_triple, valid_data_triple, test_data_triple,
                         embedding_model_dict, sw)
        end_time = time.time()
        build_time += end_time - start_time
        return build_time

    def load(self, train_data_triple, valid_data_triple, test_data_triple, sw, seed):
        initialize_random_seeds(seed)
        embedding_model_dict, build_time = self.build_embedding_models()
        start_time = time.time()
        self.build_model(train_data_triple, valid_data_triple, test_data_triple,
                         embedding_model_dict, sw, is_train=False)
        end_time = time.time()
        build_time += end_time - start_time
        return build_time

    def estimate(self, test_data, is_tqdm=True):
        model_dict = self.selectivity_model_dict
        string_helper = self.string_helper
        selectivity_learner_configs_triple = self.selectivity_learner_configs_triple
        embedding_model_dict = self.embedding_model_dict

        selectivity_learner_configs = selectivity_learner_configs_triple[0]

        normalized_preds, estimations = get_selectivity_for_strings_AstridEach(
            test_data,
            embedding_model_dict,
            model_dict,
            string_helper,
            selectivity_learner_configs,
        )

        return estimations

    def model_size(self, fn_desc=None):
        size_embed = 0
        size_model = 0
        if fn_desc is None:
            fn_desc_list = self.fn_desc_list
        else:
            fn_desc_list = [fn_desc]
        for idx, fn_desc in enumerate(fn_desc_list):
            selectivity_learner_configs = self.selectivity_learner_configs_triple[idx]
            model_path = selectivity_learner_configs.selectivity_model_file_name

            embedding_model_file_name = os.path.join(
                self.embedding_model_dirpath, f"{fn_desc}.pth"
            )
            size_embed_part = os.path.getsize(embedding_model_file_name)
            print(f"{fn_desc = }, {size_embed_part = }")
            size_embed += size_embed_part
            size_model_part = os.path.getsize(model_path)
            size_model += size_model_part

        size_total = size_model + size_embed
        print(f"{size_model = }")
        print(f"{size_embed = }")
        print(f"{size_total = }")

        return size_total
