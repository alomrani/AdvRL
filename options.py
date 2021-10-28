import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Options for training RL agent to generate attacks for training a detection model."
    )

    # Data

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi-pixel",
        help="attack mode: multi-pixel or one-pixel",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="mask previously perturbed pixels"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="combined_mal",
        help="Type of adversarial agent to train/test",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=1000.,
        help="Number of instances used for reporting validation performance",
    )

    parser.add_argument(
        "--val_dataset",
        type=str,
        default="rl_datasets/rl_val.pt",
        help="Dataset file to use for validation",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="rl_datasets/rl_test.pt",
        help="Dataset file to use for testing",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="rl_datasets/rl_train.pt",
        help="Dataset file to use for training",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=10000, help="Dataset size for training",
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=264,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=48,
        help="Number of timesteps in charging simulation",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.15, help="Learning rate decay per epoch"
    )

    parser.add_argument(
        "--delta", type=float, default=0.001, help="Learning rate decay per epoch"
    )

    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=0.001,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )

    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--exp_beta",
        type=float,
        default=0.7,
        help="Exponential moving average baseline decay (default 0.8)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="hyperparameter to control perturbation",
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Set this value to only evaluate model on a specific graph size",
    )


    parser.add_argument(
        "--eval_output", type=str, default=".", help="path to output evaulation plots",
    )
    parser.add_argument(
        "--load_path", type=str, default=None, help="path to agent's parameters",
    )
    parser.add_argument(
        "--load_path2", type=str, default="./best_model.pt", help="path to detection model's parameters",
    )
    parser.add_argument(
        "--load_paths", nargs="+", default=[], help="path to agent's parameters",
    )
    # Misc
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Set this to true if you want to tune the hyperparameters",
    )

    parser.add_argument(
        "--output_dir", default="outputs", help="Directory to write output models to"
    )

    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=0,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )


    parser.add_argument(
        "--save_dir", help="Path to save the checkpoints",
    )



    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.use_cuda:
        opts.device = "cuda"
    else:
        opts.device = "cpu"

    opts.run_name = "{}_{}".format("run", time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(opts.output_dir, opts.run_name)
    assert (
        opts.dataset_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"
    return opts
