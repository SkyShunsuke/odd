import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Flow-matching anomaly detection")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/train/mvtec_dit_fm.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=True,
        choices=["train", "eval", "eval_optimal"],
    )

    ### Loaded from torchrun
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--img_metric_only", action="store_true", help="Only evaluate image-level metrics (AUROC, AUPR, F1-max) without pixel-level metrics (PRO, AU-PRO, etc.)")
    args = parser.parse_args()
    return args

def main(params, args):
    """Dispatch the requested task.

    - train: train the flow-matching velocity field on frozen-backbone latents.
    - eval: evaluate a trained checkpoint (reconstruction / inversion / density).
    - eval_optimal: evaluate the closed-form optimal denoiser (no checkpoint).
    """
    task = args.task
    if task == "train":
        from src.vfad.train import main as vfad_train
        vfad_train(params, args)
    elif task == "eval":
        from src.vfad.eval import main as vfad_eval
        vfad_eval(params, args)
    elif task == "eval_optimal":
        from src.vfad.eval_optimal_denoiser import main as vfad_eval_optimal
        vfad_eval_optimal(params, args)
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    main(params, args)
