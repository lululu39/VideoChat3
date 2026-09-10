import argparse

import torch.distributed as dist

from xtuner.v1.train.videochat3_curriculum import VideoChat3CurriculumTrainer
from xtuner.v1.utils import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    trainer = VideoChat3CurriculumTrainer.from_config(Config.fromfile(args.config)["trainer"])
    trainer.fit()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
