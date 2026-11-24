"""Visualize SAVi slot attention decomposition.

Loads a trained SAVi/StoSAVi checkpoint, runs it on validation samples,
and saves a grid image per sample showing:
  original frames | combined reconstruction | per-slot masked reconstructions

Usage:
    python scripts/visualize_savi_slots.py \
        --params slotformer/bev_slots/configs/savi_bev_params.py \
        --weight /path/to/checkpoint.pth \
        --dataset_path /path/to/data \
        --save_dir ./slot_visualizations \
        --n_samples 5
"""

import os
import sys
import csv
import json
from pathlib import Path
import random
import argparse
import importlib

import numpy as np
import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image, ImageFile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ImageFile.LOAD_TRUNCATED_IMAGES = True

# add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slotformer.bev_slots.models import build_model


def valid_starts(folder, n_frames, frame_offset):
    frames = {int(p.stem) for p in Path(folder).glob('*.jpg') if p.stem.isdigit()}
    return [start for start in sorted(frames)
            if all(start + t * frame_offset in frames for t in range(n_frames))]


def load_video_frames(folder, transform, frame_ids):
    frames = []
    for idx in frame_ids:
        with Image.open(Path(folder) / f'{idx:04d}.jpg') as image:
            frames.append(transform(image.convert('RGB')))
    return torch.stack(frames, dim=0)


def make_video_grid(imgs, recon_combined, recons, masks):
    """Make a grid showing original | reconstruction | per-slot decomposition.

    Args:
        imgs: [T, C, H, W]
        recon_combined: [T, C, H, W]
        recons: [T, num_slots, C, H, W]
        masks: [T, num_slots, 1, H, W]
    """
    out = (
        torch.cat([
            imgs.unsqueeze(1),
            recon_combined.unsqueeze(1),
            recons * masks + (1. - masks),  # masked slots on white bg
        ], dim=1)
    )  # [T, num_slots+2, 3, H, W]

    grid_per_frame = [
        vutils.make_grid(out[t], nrow=out.shape[1], pad_value=1.)
        for t in range(imgs.shape[0])
    ]
    return grid_per_frame  # list of [3, H, W'] tensors


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Visualize SAVi slot decomposition')
    parser.add_argument('--params', type=str, required=True,
                        help='Path to params file')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--save_dir', type=str, default='./slot_visualizations',
                        help='Directory to save output images')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of videos to visualize')
    parser.add_argument('--n_frames', type=int, default=6,
                        help='Number of frames per video')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible frame selection')
    parser.add_argument('--dataset_path', '--dataset-path', required=True,
                        help='Mounted Obj3D dataset root containing test/')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to sample from')
    args = parser.parse_args()

    if args.n_samples <= 0 or args.n_frames <= 0:
        parser.error('n_samples and n_frames must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load params
    params_path = args.params
    if params_path.endswith('.py'):
        params_path = params_path[:-3]
    sys.path.append(os.path.dirname(params_path))
    params_mod = importlib.import_module(os.path.basename(params_path))
    params = params_mod.SlotFormerParams()

    # build and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(params)
    ckp = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(ckp.get('state_dict', ckp), strict=True)
    model.testing = False  # we need decoding for reconstructions + masks
    model = model.to(device).eval()

    # Match the training crop/resize and [0, 1] pixel range.
    spatial = (transforms.CenterCrop(params.resolution) if params.crop_bevs
               else transforms.Resize(params.resolution))
    transform = transforms.Compose([spatial, transforms.ToTensor()])
    data_root = Path(args.dataset_path) / args.split
    if not data_root.is_dir():
        raise FileNotFoundError(f'Missing dataset split: {data_root}')
    if params.frame_offset <= 0:
        raise ValueError('frame_offset must be positive')
    eligible = []
    for folder in sorted(data_root.iterdir()):
        if folder.is_dir():
            starts = valid_starts(folder, args.n_frames, params.frame_offset)
            if starts:
                eligible.append((folder, starts))
    if args.n_samples > len(eligible):
        raise ValueError(f'Requested {args.n_samples} samples but only '
                         f'{len(eligible)} videos contain a complete clip')
    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.n_samples)
    manifest = []
    for folder, starts in selected:
        start = rng.choice(starts)
        manifest.append(dict(folder=str(folder), frames=[
            start + t * params.frame_offset for t in range(args.n_frames)]))
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir) / 'samples.json', 'w') as f:
        json.dump(dict(weight=args.weight, seed=args.seed, split=args.split,
                       samples=manifest), f, indent=2)
    n_samples = len(manifest)

    all_mse, all_psnr, all_ssim = [], [], []

    for i, sample in enumerate(manifest):
        folder = sample['folder']
        video_name = os.path.basename(folder)
        print(f'[{i+1}/{n_samples}] Processing {video_name}...')

        # load frames
        video = load_video_frames(
            folder, transform, sample['frames']
        ).to(device)  # [T, C, H, W]

        # run model
        in_dict = {'img': video[None]}  # [1, T, C, H, W]
        out_dict = model(in_dict)

        # extract outputs, remove batch dim
        recon_combined = out_dict['post_recon_combined'][0]  # [T, C, H, W]
        recons = out_dict['post_recons'][0]  # [T, num_slots, C, H, W]
        masks = out_dict['post_masks'][0]  # [T, num_slots, 1, H, W]

        imgs = video.type_as(recon_combined)

        # compute reconstruction quality metrics
        x = imgs.cpu().numpy()
        y = recon_combined.cpu().numpy()
        vid_mse = ((x - y) ** 2).mean()
        vid_psnr = np.mean([
            peak_signal_noise_ratio(x[t], y[t], data_range=1.)
            for t in range(x.shape[0])
        ])
        vid_ssim = np.mean([
            structural_similarity(
                x[t] * 255., y[t] * 255.,
                channel_axis=0, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False, data_range=255,
            )
            for t in range(x.shape[0])
        ])
        all_mse.append(vid_mse)
        all_psnr.append(vid_psnr)
        all_ssim.append(vid_ssim)
        print(f'  MSE: {vid_mse:.4f}  PSNR: {vid_psnr:.2f}  SSIM: {vid_ssim:.4f}')

        # save per-frame: original and reconstruction side by side
        for t in range(imgs.shape[0]):
            pair = torch.stack([imgs[t].cpu(), recon_combined[t].cpu()], dim=0)
            pair_grid = vutils.make_grid(pair, nrow=2, pad_value=1.)
            save_path = os.path.join(
                args.save_dir, f'{video_name}_recon_frame{t:02d}.png')
            vutils.save_image(pair_grid, save_path)

        # save per-slot images for the last frame: original | single slot
        last = args.n_frames - 1
        last_img = imgs[last].cpu()  # [C, H, W]
        last_recons = recons[last].cpu()  # [num_slots, C, H, W]
        last_masks = masks[last].cpu()  # [num_slots, 1, H, W]
        for s in range(last_recons.shape[0]):
            slot_vis = last_recons[s] * last_masks[s] + (1. - last_masks[s])
            pair = torch.stack([last_img, slot_vis], dim=0).clamp(0, 1)
            pair_grid = vutils.make_grid(pair, nrow=2, pad_value=1.)
            save_path = os.path.join(
                args.save_dir, f'{video_name}_slot{s:02d}.png')
            vutils.save_image(pair_grid, save_path)

        # save slot decomposition grid per frame
        frame_grids = make_video_grid(imgs, recon_combined, recons, masks)
        for t, grid in enumerate(frame_grids):
            save_path = os.path.join(
                args.save_dir, f'{video_name}_slots_frame{t:02d}.png')
            vutils.save_image(grid, save_path)

        # also save a vertical stack of all frames
        all_frames = torch.stack(frame_grids, dim=0)  # [T, 3, H, W']
        stack = vutils.make_grid(all_frames, nrow=1, pad_value=1.)
        save_path = os.path.join(args.save_dir, f'{video_name}_slots_all.png')
        vutils.save_image(stack, save_path)

        print(f'  Saved to {args.save_dir}/{video_name}_*.png')

    # write per-video CSV
    csv_path = os.path.join(args.save_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'mse', 'psnr', 'ssim'])
        for sample, mse, psnr, ssim in zip(
                manifest, all_mse, all_psnr, all_ssim):
            video_name = os.path.basename(sample["folder"])
            writer.writerow([video_name, f'{mse:.6f}', f'{psnr:.4f}', f'{ssim:.6f}'])
        # summary row
        writer.writerow([
            'MEAN', f'{np.mean(all_mse):.6f}',
            f'{np.mean(all_psnr):.4f}', f'{np.mean(all_ssim):.6f}',
        ])
        writer.writerow([
            'STD', f'{np.std(all_mse):.6f}',
            f'{np.std(all_psnr):.4f}', f'{np.std(all_ssim):.6f}',
        ])
    print(f'\nMetrics saved to {csv_path}')

    # print summary
    print(f'\n{"="*50}')
    print(f'Reconstruction quality ({n_samples} videos, {args.split} split):')
    print(f'  MSE:  {np.mean(all_mse):.4f} (+/- {np.std(all_mse):.4f})')
    print(f'  PSNR: {np.mean(all_psnr):.2f} (+/- {np.std(all_psnr):.2f})')
    print(f'  SSIM: {np.mean(all_ssim):.4f} (+/- {np.std(all_ssim):.4f})')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
