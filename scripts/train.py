"""A unified training script for all models used in the SlotFormer project."""

import os
import sys
import importlib
import argparse
import wandb
from pprint import pprint
import torch

from nerv.utils import mkdir_or_exist
from nerv.training import BaseDataModule


def main(params):
    # build datamodule
    datasets = build_dataset(params)
    train_set, val_set = datasets[0], datasets[1]

    collate_fn = datasets[2] if len(datasets) == 3 else None
    datamodule = BaseDataModule(
        params,
        train_set=train_set,
        val_set=val_set,
        use_ddp=params.ddp,
        collate_fn=collate_fn,
    )

    

    # build model
    model = build_model(params)

    exp_name = os.path.splitext(os.path.basename(args.params))[0]
    ckp_path = os.path.join(args.output_dir, exp_name, 'models')
    if args.local_rank == 0:
        mkdir_or_exist(ckp_path)
        wandb.init(project=params.project, name=exp_name, dir=ckp_path)

    method = build_method(
        model=model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=args.local_rank,
        use_ddp=args.ddp,
        use_fp16=args.fp16,
    )


    method.fit(
        resume_from=args.weight, san_check_val_step=params.san_check_val_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SlotFormer training script')
    parser.add_argument('--task', type=str, default='bev_slots')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--dataset_path', '--dataset-path', type=str, default=None, help='dataset path for training')
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true', help='half-precision')
    parser.add_argument('--ddp', action='store_true', help='DDP training')
    parser.add_argument('--cudnn', action='store_true', help='cudnn benchmark')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--output-dir', default='outputs/pretraining')
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

    # import `build_dataset/model/method` function according to `args.task`
    print(f'INFO: training model in {args.task} task!')
    task = importlib.import_module(f'slotformer.{args.task}')
    build_dataset = task.build_dataset
    build_model = task.build_model
    build_method = task.build_method

    # load the params
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    params.data_root = args.dataset_path if args.dataset_path else None
    params.ddp = args.ddp
    if args.ddp:
        params.gpus = int(os.environ.get("WORLD_SIZE", params.gpus))

    if args.local_rank == 0:
        print("\n========== PARAMETERS ==========")

        param_dict = {
            name: getattr(params, name)
            for name in dir(params)
            if not name.startswith("_")
            and not callable(getattr(params, name))
        }

        pprint(param_dict, sort_dicts=True)

        print("================================\n")

    if args.fp16:
        print('INFO: using FP16 training!')
    if args.ddp:
        print('INFO: using DDP training!')
    if args.cudnn:
        torch.backends.cudnn.benchmark = True
        print('INFO: using cudnn benchmark!')

    main(params)
