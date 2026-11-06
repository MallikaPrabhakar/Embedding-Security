"""
Train a diffusion model on images.
"""

import argparse
import time
import os

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    os.environ["DEVICE_ID"] = args.device_id

    dist_util.setup_dist()
    if args.checkpoint_dir:
        log_prefix=f"{args.eval_dset}_" if args.eval_dset != "" else ""
        logger.configure(args.checkpoint_dir, log_prefix=log_prefix)
    else:
        logger.configure()
    logger.log("using device ", dist_util.dev())

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        emb_dir=args.emb_dir,
    )

    logger.log("training...")
    start_time = time.time()
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        classifier_free=args.classifier_free,
    ).run_loop()
    end_time = time.time()
    time_elapsed = end_time - start_time
    logger.log(f"Time elapsed {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )


def create_argparser():
    defaults = dict(
        data_dir="",
        emb_dir=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        checkpoint_dir=None,
        classifier_free=False,
        device_id="0",
        eval_dset="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
