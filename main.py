import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import gc
import logging
import os
from datetime import datetime

from utils.utils import seed_everything, print_to_list
from utils.load_data import load_data
import models
from train import Alchemist


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    dataset_cfg, model_cfg = cfg.data, cfg.model

    # get train and validation data
    train_gen, valid_gen, test_gen, feature_map = load_data(dataset_cfg)
    model_class = getattr(models, model_cfg.model)
    model = model_class(feature_map, **OmegaConf.to_container(model_cfg))
    logging.info(f"Total number of parameters: {model.count_parameters()}")

    # train the model
    # if len(cfg.gpu) == 1:
    #     gpu = cfg.gpu[0]
    # else:
    gpu = cfg.gpu[HydraConfig.get().job.get('num', 0) % len(cfg.gpu)]
    logging.info(f"Use GPU {gpu}")
    alchemist = Alchemist(model, gpu)
    logdir = '.'
    alchemist.train(train_gen, valid_gen, cfg, logdir)

    # get evaluation results on validation
    logging.info('****** Validation evaluation ******')
    valid_result = alchemist.evaluate(valid_gen, cfg.metrics)
    logging.info('[Best Val Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in valid_result.items()))
    del train_gen, valid_gen
    gc.collect()

    # get evaluation results on test
    logging.info('******** Test evaluation ********')
    test_result = alchemist.evaluate(test_gen, cfg.metrics)
    logging.info('[Best Test Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in test_result.items()))

    # save the results to csv
    result_file = "result.csv"
    logging.info('Save results to {}'.format(os.path.abspath(result_file)))

    with open(result_file, 'a+') as fw:
        fw.write(
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")},[dataset_name] {dataset_cfg.name},'
            f' [val] {print_to_list(valid_result)}, [test] {print_to_list(test_result)}\n')


if __name__ == "__main__":
    my_app()
