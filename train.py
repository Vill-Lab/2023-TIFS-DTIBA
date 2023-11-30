
import sys

sys.path.append('.')
import argparse
import os
import evaluation

from identity_hushing.fastreid.config import get_cfg
from identity_hushing.fastreid.engine import DefaultTrainer,default_setup
from identity_hushing.fastreid.utils.checkpoint import Checkpointer

from identity_hushing.fastreid.data import build_reid_test_loader

def default_argument_parser():
    """
    Create a parser with some common arguments used by fastreid users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="identity_hushing/configs/Market1501/sbs_R50-ibn.yml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def train_victim_model():
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    normal_loader, normal_query=build_reid_test_loader(cfg, dataset_name='Market1501')
    poison_loader, poison_query=build_reid_test_loader(cfg, dataset_name='Market1501',clean_set='out/')
    clean_r10,ASR=evaluation.val_atk(trainer.model,normal_query,normal_loader,poison_query, poison_loader)
    print("Rank-10 accuracy for clean samples", clean_r10)
    print("ASR for poisoned samples", ASR)

