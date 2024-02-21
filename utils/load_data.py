import os
from hydra.utils import get_original_cwd, to_absolute_path
from utils.features import FeatureMap, FeatureEncoder
from utils import datasets
import logging


def load_data(dataset_cfg):
    """Load dataset from csv or h5 file.
    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Directory of data.
        data_format (str): Data format, csv or h5.
        **kwargs: Other arguments.
    Returns:
        train_gen: Training data generator.
        valid_gen: Validation data generator.
        test_gen: Test data generator.
        feature_map: Feature map.
    """
    dataset_cfg.data_root = to_absolute_path(dataset_cfg.data_root)
    dataset_cfg.train_data = to_absolute_path(dataset_cfg.train_data)
    dataset_cfg.valid_data = to_absolute_path(dataset_cfg.valid_data)
    dataset_cfg.test_data = to_absolute_path(dataset_cfg.test_data)

    data_dir = os.path.join(dataset_cfg.data_root, dataset_cfg.name)

    if dataset_cfg.data_format == 'h5':
        feature_map = FeatureMap(dataset_cfg.name, data_dir)
        json_file = os.path.join(data_dir, "feature_map.json")
        if os.path.exists(json_file):
            feature_map.load(json_file)
        else:
            raise RuntimeError('feature_map not exist!')
    else:  # load data from csv
        dataset = dataset_cfg.name.split('_')[0].lower()
        try:
            feature_encoder = getattr(datasets, dataset).FeatureEncoder(**dataset_cfg)
        except:
            feature_encoder = FeatureEncoder(**dataset_cfg)

        if os.path.exists(feature_encoder.json_file):
            feature_encoder.feature_map.load(feature_encoder.json_file)
        else:
            datasets.build_dataset(feature_encoder, **dataset_cfg)

        dataset_cfg.train_data = os.path.join(data_dir, 'train*.h5')
        dataset_cfg.valid_data = os.path.join(data_dir, 'valid*.h5')
        dataset_cfg.test_data = os.path.join(data_dir, 'test*.h5')
        feature_map = feature_encoder.feature_map

    train_gen, valid_gen = datasets.h5_generator(feature_map, stage='train', **dataset_cfg)
    test_gen = datasets.h5_generator(feature_map, stage='test', **dataset_cfg)

    number_of_fields = len(feature_map.feature_specs)
    number_of_features = 0

    for field, field_spec in feature_map.feature_specs.items():
        num_of_features_in_field = field_spec.get("vocab_size", None)
        number_of_features += num_of_features_in_field

    logging.info(f'Number of fields: {number_of_fields}')
    logging.info(f'Number of features: {number_of_features}')

    return train_gen, valid_gen, test_gen, feature_map