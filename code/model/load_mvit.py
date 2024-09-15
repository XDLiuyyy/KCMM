from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu

import torch


def get_mvit_model():
    config_file_path = '/home/ly/CAction/KCMM/code/model/configs/Kinetics/MVITv2_S_16x4.yaml'
    cfg = load_config(config_file_path)
    cfg = assert_and_infer_cfg(cfg)
    cfg.NUM_GPUS = 1
    cfg.TRAIN.CHECKPOINT_FILE_PATH = '/home/ly/CAction/KCMM/code/model/pretrained_weights/MViTv2_S_16x4_k400_f302660347.pyth'

    model = build_model(cfg)

    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        print("Load from given checkpoint file:{}".format(cfg.TRAIN.CHECKPOINT_FILE_PATH))
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            False,
            None,
            None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )

    return model

def Net():
    net = get_mvit_model()

    return net

