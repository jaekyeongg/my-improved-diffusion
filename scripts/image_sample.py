"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision as tv
import time

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


#manual_model_path = None
"""
manual_model_path = ["./ema_0.9999_010000.pt",
"./ema_0.9999_020000.pt","./ema_0.9999_030000.pt",
"./ema_0.9999_040000.pt","./ema_0.9999_050000.pt",
"./ema_0.9999_060000.pt","./ema_0.9999_070000.pt",
"./ema_0.9999_080000.pt","./ema_0.9999_090000.pt",
"./ema_0.9999_100000.pt","./ema_0.9999_110000.pt",
"./ema_0.9999_120000.pt","./ema_0.9999_130000.pt",
"./ema_0.9999_140000.pt","./ema_0.9999_150000.pt",
"./ema_0.9999_160000.pt","./ema_0.9999_170000.pt",
"./ema_0.9999_180000.pt","./ema_0.9999_190000.pt",
"./ema_0.9999_200000.pt","./ema_0.9999_210000.pt",
"./ema_0.9999_220000.pt","./ema_0.9999_230000.pt",
"./ema_0.9999_240000.pt","./ema_0.9999_250000.pt",
"./ema_0.9999_260000.pt","./ema_0.9999_270000.pt",
"./ema_0.9999_280000.pt","./ema_0.9999_290000.pt",
"./ema_0.9999_300000.pt"
]
"""
manual_model_path = [
"./ema_0.9999_180000.pt","./ema_0.9999_190000.pt",
"./ema_0.9999_200000.pt","./ema_0.9999_210000.pt",
"./ema_0.9999_220000.pt","./ema_0.9999_230000.pt",
"./ema_0.9999_240000.pt","./ema_0.9999_250000.pt",
"./ema_0.9999_260000.pt","./ema_0.9999_270000.pt",
"./ema_0.9999_280000.pt","./ema_0.9999_290000.pt",
"./ema_0.9999_300000.pt"
]
seed = 20
def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print("seed : " , seed)
    if manual_model_path is None :
        print("args model path!! " + args.model_path)
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        logger.log("sampling...")
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            print("model inference start png")
            th.manual_seed(seed)
            th.cuda.manual_seed(seed)
            input_noise = th.randn(size = [args.batch_size, 3, args.image_size, args.image_size], device = 'cuda')
            tv.utils.save_image(input_noise, args.model_path + "_" + str(seed) + "_input_noise.png", nrow=8, normalize = True, range=(-1,1))
            s = time.time()
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                noise = input_noise
            )
            print("sample shape : ", sample.shape)
            tv.utils.save_image(sample, args.model_path  + "_" + str(seed) + "_sample.png", nrow=8, normalize = True, range=(-1,1))
            print("model inference end ")
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

    else :
        for model_path in manual_model_path :
            print("manual model path!! " + model_path)
            model.load_state_dict(
                dist_util.load_state_dict(model_path, map_location="cpu")
            )
            model.to(dist_util.dev())
            model.eval()

            logger.log("sampling...")
            all_images = []
            all_labels = []
    
            while len(all_images) * args.batch_size < args.num_samples:
                model_kwargs = {}
                if args.class_cond:
                    classes = th.randint(
                        low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                    )
                    model_kwargs["y"] = classes
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                print("model inference start png")
                th.manual_seed(seed)
                th.cuda.manual_seed(seed)
                input_noise = th.randn(size = [args.batch_size, 3, args.image_size, args.image_size], device = 'cuda')
                tv.utils.save_image(input_noise, model_path + "_" + str(seed) + "_input_noise.png", nrow=8, normalize = True, range=(-1,1))
                s = time.time()
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    noise = input_noise
                )
                print("sample shape : ", sample.shape)
                tv.utils.save_image(sample, model_path  + "_" + str(seed) + "_sample.png", nrow=8, normalize = True, range=(-1,1))
                print("model inference end ")
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
   
        
        
        
        
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #sample = sample.permute(0, 2, 3, 1)
        #sample = sample.contiguous()
      

        #gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        #dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        #all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        #if args.class_cond:
        #    gathered_labels = [
        #        th.zeros_like(classes) for _ in range(dist.get_world_size())
        #    ]
        #    dist.all_gather(gathered_labels, classes)
        #    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        #logger.log(f"created {len(all_images) * args.batch_size} samples")

    #arr = np.concatenate(all_images, axis=0)
    #arr = arr[: args.num_samples]
    #if args.class_cond:
    #    label_arr = np.concatenate(all_labels, axis=0)
    #    label_arr = label_arr[: args.num_samples]
    #if dist.get_rank() == 0:
    #    shape_str = "x".join([str(x) for x in arr.shape])
    #    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #    logger.log(f"saving to {out_path}")
    #    if args.class_cond:
    #        np.savez(out_path, arr, label_arr)
    #    else:
    #        np.savez(out_path, arr)

    #dist.barrier()
    #logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
