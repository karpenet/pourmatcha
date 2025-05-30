#!/usr/bin/env python

"""
Custom training script with image augmentation enabled.
This script sets up image transforms inline and runs ACT training.
"""

import logging
from lerobot.common.datasets.transforms import (
    ImageTransformsConfig,
    ImageTransformConfig
)
import lerobot
from lerobot.configs.default import DatasetConfig
from lerobot.common.policies.act.configuration_act import ACTConfig 
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import WandBConfig
from lerobot.scripts.train import train
from lerobot.common.utils.utils import init_logging
import time
import os
from pathlib import Path
os.environ['WANDB_NOTEBOOK_NAME'] = 'train_act'


def main():
    # Initialize logging
    init_logging()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Configure image transforms - enable all except RandomResizedCrop
    image_transforms_config = ImageTransformsConfig(
        enable=True,
        max_num_transforms=6,  # All transforms except crop_scale
        random_order=False,
        tfs={
            # Disable RandomResizedCrop by setting weight to 0
            "crop_scale": ImageTransformConfig(
                weight=0.0,
                type="RandomResizedCrop",
                kwargs={
                    "size": (224, 224),
                    "scale": (0.8, 1.0),
                    "ratio": (0.75, 1.33),
                }
            ),
            # Keep all other transforms enabled
            "rotate": ImageTransformConfig(
                weight=1.0,
                type="RandomRotation",
                kwargs={"degrees": 30}
            ),
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)}
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)}
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)}
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)}
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)}
            ),
        }
    )

    # Configure dataset
    dataset_config = DatasetConfig(
        repo_id="jchun/so100_cleaning_merge",
        image_transforms=image_transforms_config,
        use_imagenet_stats=True,
    )

    # Configure ACT policy
    policy_config = ACTConfig(
        device="cuda",
    )

    # Configure training pipeline
    output_dir = Path(f"/workspace/outputs/train/act_so100_cleaning/{timestamp}")
    print("save to", output_dir)
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=output_dir,
        job_name="act_so100_cleaning",
        wandb=WandBConfig(True),
        seed=42,
        batch_size=12,
        steps=100000,
        eval_freq=5000,
        log_freq=50,
        save_freq=1000,
        save_checkpoint=True,
    )

    logging.info("Starting training with image augmentation "
                 "(all transforms except RandomResizedCrop)")
    enabled_transforms = [
        transform_name
        for transform_name, config in image_transforms_config.tfs.items()
        if config.weight > 0
    ]
    logging.info(f"Enabled transforms: {enabled_transforms}")

    # Run training
    train(train_config)


main()
