# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1, 2022/7/6
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan, Gaowei Zhang
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn, zgw15630559577@163.com

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os
import pickle
import warnings
from typing import Literal

from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils.argument_list import dataset_arguments


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("recbole.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )
    return train_data, valid_data, test_data


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    # 检查是否启用k样本抽取功能
    if config.get("enable_k_sampling", False):
        k = config.get("k_samples", 1000)
        return _create_k_sample_dataloaders(config, dataset, k)
    
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        valid_data = get_dataloader(config, "valid")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data


def _create_k_sample_dataloaders(config, dataset, k):
    """Create dataloaders with k positive and k negative samples.
    
    Args:
        config (Config): Configuration object.
        dataset (Dataset): Original dataset.
        k (int): Number of positive and negative samples to extract.
        
    Returns:
        tuple: (train_data, valid_data, test_data) dataloaders with k samples each.
    """
    import torch
    import numpy as np
    from recbole.utils import getLogger, set_color
    
    logger = getLogger()
    logger.info(set_color(f"Creating k-sample dataloaders with k={k}", "pink"))
    
    # 构建原始数据集
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    
    # 为每个数据集创建k样本版本
    train_k_dataset = _extract_k_samples(train_dataset, dataset, k, "train")
    valid_k_dataset = _extract_k_samples(valid_dataset, dataset, k, "valid") 
    test_k_dataset = _extract_k_samples(test_dataset, dataset, k, "test")
    
    # 创建对应的dataloader
    train_data = get_dataloader(config, "train")(
        config, train_k_dataset, None, shuffle=config["shuffle"]
    )
    valid_data = get_dataloader(config, "valid")(
        config, valid_k_dataset, None, shuffle=False
    )
    test_data = get_dataloader(config, "test")(
        config, test_k_dataset, None, shuffle=False
    )
    
    logger.info(set_color(f"K-sample dataloaders created successfully", "green"))
    return train_data, valid_data, test_data


def _extract_k_samples(dataset, original_dataset, k, phase):
    """Extract k positive and k negative samples from dataset.
    
    Args:
        dataset: The split dataset (train/valid/test).
        original_dataset: The original complete dataset.
        k (int): Number of samples to extract.
        phase (str): Phase name for logging.
        
    Returns:
        Dataset: New dataset with k positive and k negative samples.
    """
    import torch
    import numpy as np
    from recbole.utils import getLogger, set_color
    
    logger = getLogger()
    
    # 获取正样本（现有交互）
    pos_interactions = dataset.dataset
    pos_count = len(pos_interactions)
    
    if pos_count < k:
        logger.warning(f"Phase {phase}: Only {pos_count} positive samples available, using all.")
        k_pos = pos_count
    else:
        k_pos = k
    
    # 随机选择k个正样本
    pos_indices = np.random.choice(pos_count, k_pos, replace=False)
    selected_pos = pos_interactions[pos_indices]
    
    # 生成负样本
    neg_samples = _generate_negative_samples(original_dataset, selected_pos, k)
    
    # 合并正负样本
    if len(neg_samples) > 0:
        # 为负样本添加标签（如果需要）
        combined_data = torch.cat([selected_pos, neg_samples], dim=0)
        
        # 创建标签：正样本为1，负样本为0
        pos_labels = torch.ones(len(selected_pos))
        neg_labels = torch.zeros(len(neg_samples))
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        logger.info(set_color(f"Phase {phase}: Created {len(selected_pos)} positive and {len(neg_samples)} negative samples", "cyan"))
    else:
        combined_data = selected_pos
        labels = torch.ones(len(selected_pos))
        logger.info(set_color(f"Phase {phase}: Created {len(selected_pos)} positive samples only", "cyan"))
    
    # 创建新的数据集对象
    new_dataset = copy.deepcopy(dataset)
    new_dataset.dataset = combined_data
    
    # 如果数据集支持标签，添加标签信息
    if hasattr(new_dataset, 'labels'):
        new_dataset.labels = labels
    
    return new_dataset


def _generate_negative_samples(dataset, positive_samples, k):
    """Generate k negative samples based on positive samples.
    
    Args:
        dataset: Original dataset containing all items and users.
        positive_samples: Selected positive interactions.
        k (int): Number of negative samples to generate.
        
    Returns:
        torch.Tensor: Negative samples tensor.
    """
    import torch
    import numpy as np
    
    # 获取所有用户和物品ID
    all_users = dataset.user_token2id.values() if hasattr(dataset, 'user_token2id') else range(dataset.user_num)
    all_items = dataset.item_token2id.values() if hasattr(dataset, 'item_token2id') else range(dataset.item_num)
    
    # 创建正样本的用户-物品对集合（用于避免重复）
    pos_pairs = set()
    user_col = dataset.uid_field
    item_col = dataset.iid_field
    
    for interaction in positive_samples:
        user_id = interaction[dataset.field2id_token[user_col]].item()
        item_id = interaction[dataset.field2id_token[item_col]].item()
        pos_pairs.add((user_id, item_id))
    
    # 生成负样本
    neg_samples = []
    attempts = 0
    max_attempts = k * 10  # 避免无限循环
    
    while len(neg_samples) < k and attempts < max_attempts:
        # 随机选择用户和物品
        user_id = np.random.choice(list(all_users))
        item_id = np.random.choice(list(all_items))
        
        # 确保不是正样本
        if (user_id, item_id) not in pos_pairs:
            # 创建负样本交互
            neg_interaction = torch.zeros_like(positive_samples[0])
            neg_interaction[dataset.field2id_token[user_col]] = user_id
            neg_interaction[dataset.field2id_token[item_col]] = item_id
            
            neg_samples.append(neg_interaction)
            pos_pairs.add((user_id, item_id))  # 避免重复生成
        
        attempts += 1
    
    if len(neg_samples) == 0:
        return torch.empty(0, positive_samples.size(1))
    
    return torch.stack(neg_samples)


def _get_AE_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                distribution,
                alpha,
            )
    return sampler


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """ 
    train_neg_sample_args = config["train_neg_sample_args"]
    valid_neg_sample_args = config["valid_neg_sample_args"]
    test_neg_sample_args = config["test_neg_sample_args"]
    repeatable = config["repeatable"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
    )
    train_sampler = base_sampler.set_phase("train") if base_sampler else None
    
    valid_sampler = _create_sampler(
        dataset,
        built_datasets,
        valid_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    valid_sampler = valid_sampler.set_phase("valid") if valid_sampler else None

    test_sampler = _create_sampler(
        dataset,
        built_datasets,
        test_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    test_sampler = test_sampler.set_phase("test") if test_sampler else None
    return train_sampler, valid_sampler, test_sampler

def add_negs(config, dataset, built_datasets, sampler):
    user_ids = dataset["user_id"]
    value_ids = sampler.sample_by_key_ids(user_ids, 1)