"""Per-rank loss-spike captures and distributed training diagnostics."""
import json
import math
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import median

import torch
import wandb
from torchvision.utils import save_image


class TrainingDiagnostics:
    def __init__(self, method):
        self.method = method
        self.history = deque(maxlen=getattr(method.params, 'loss_spike_window', 100))
        self.factor = getattr(method.params, 'loss_spike_factor', 2.0)
        self.min_delta = getattr(method.params, 'loss_spike_min_delta', 0.05)
        self.warmup = getattr(method.params, 'loss_spike_warmup', 20)
        self.cooldown = getattr(method.params, 'loss_spike_cooldown', 100)
        self.last_capture = -self.cooldown
        run = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        self.root = Path(method.params.loss_spike_root) / run
        self.context = None

    @staticmethod
    def distributed():
        return torch.distributed.is_initialized()

    def reduce(self, tensor, op=torch.distributed.ReduceOp.MAX):
        tensor = tensor.detach().clone()
        if self.distributed():
            torch.distributed.all_reduce(tensor, op=op)
        return tensor

    def inspect(self, data, outputs, losses):
        m = self.method
        metrics = {name: value.detach().float() for name, value in losses.items()
                   if isinstance(value, torch.Tensor)}
        metrics['weighted_pixel'] = metrics['post_pixel'] * m.model.module.pixel_weight * m.params.post_recon_loss_w
        metrics['weighted_perceptual'] = metrics['post_perc_weighted'] * m.params.post_recon_loss_w
        metrics['weighted_kl'] = metrics['kld_loss'] * m.params.kld_loss_w
        kernel = outputs['kernel_dist'].detach().float()
        size = m.model.module.slot_size
        tensors = {'input': data['img'], 'mean': kernel[..., :size],
                   'logvar': kernel[..., size:], 'slots': outputs['post_slots'],
                   'reconstruction': outputs['post_recon_combined']}
        finite = torch.ones((), device=kernel.device, dtype=torch.bool)
        for name, value in tensors.items():
            value = value.detach().float()
            finite &= torch.isfinite(value).all()
            metrics[name + '_absmax'] = value.abs().max()
        metrics['logvar_min'] = kernel[..., size:].min()
        metrics['logvar_max'] = kernel[..., size:].max()
        for value in metrics.values():
            finite &= torch.isfinite(value).all()
        baseline = median(self.history) if self.history else None
        current = float(losses['loss'].detach())
        spike = (len(self.history) >= self.warmup and math.isfinite(current)
                 and current > self.factor * max(baseline, 1e-12)
                 and current - baseline > self.min_delta)
        self.context = (data, outputs, metrics, baseline)
        bad = not bool(finite)
        global_bad = bool(self.reduce((~finite).int()))
        if bad or (spike and m.it - self.last_capture >= self.cooldown):
            self.capture('nonfinite_forward' if bad else 'loss_spike')
        names = sorted(metrics)
        packed = torch.stack([metrics[name] for name in names])
        maxima = self.reduce(packed)
        minimum = self.reduce(metrics['logvar_min'], torch.distributed.ReduceOp.MIN)
        spike_any = self.reduce(torch.tensor(int(spike), device=kernel.device))
        ratio = current / max(baseline, 1e-12) if baseline is not None else 1.0
        ratio_max = self.reduce(torch.tensor(ratio, device=kernel.device))
        if m.local_rank == 0:
            logged = {f'diagnostics/{name}_rank_max': float(value)
                      for name, value in zip(names, maxima)}
            logged['diagnostics/logvar_rank_min'] = float(minimum)
            logged['diagnostics/loss_spike'] = int(spike_any)
            logged['diagnostics/loss_to_baseline_ratio_rank_max'] = float(ratio_max)
            wandb.log(logged, step=m.it)
        if global_bad:
            raise FloatingPointError('Nonfinite forward/loss detected; offending ranks saved diagnostics before backward.')
        # Do not let an isolated spike immediately raise its own reference level.
        if not spike:
            self.history.append(current)

    def check_gradient(self, norm):
        bad = not bool(torch.isfinite(norm))
        global_bad = bool(self.reduce((~torch.isfinite(norm)).int()))
        maximum = self.reduce(norm)
        if self.method.local_rank == 0:
            wandb.log({'diagnostics/gradient_norm_before_clip': float(maximum)}, step=self.method.it)
        if bad:
            self.capture('nonfinite_gradient', gradient_norm=float(norm))
        if global_bad:
            raise FloatingPointError('Nonfinite gradient detected; stopped before optimizer update.')

    def capture(self, reason, **extra):
        m = self.method
        data, outputs, metrics, baseline = self.context
        rank = torch.distributed.get_rank() if self.distributed() else 0
        folder = self.root / f'step-{m.it:08d}-rank-{rank}-{reason}'
        folder.mkdir(parents=True, exist_ok=True)
        indices = data['data_idx'].detach().cpu().tolist()
        dataset = m.train_loader.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        samples = []
        for i, index in enumerate(indices):
            route, start = dataset.valid_idx[index % len(dataset)]
            samples.append({'batch_index': i, 'data_idx': index, 'route': route,
                            'pixel_mse': float((outputs['post_recon_combined'][i].detach().float()
                                                - data['img'][i].detach().float()).square().mean()),
                            'frames': [start + t * dataset.frame_offset
                                       for t in range(data['img'].shape[1])]})
        metadata = dict(reason=reason, step=m.it, epoch=m.epoch, rank=rank,
                        lr=[group['lr'] for group in m.optimizer.param_groups],
                        baseline=baseline, samples=samples,
                        metrics={k: float(v) for k, v in metrics.items()}, **extra)
        # Use null for nonfinite numbers, keeping the metadata valid JSON.
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value
        (folder / 'diagnostics.json').write_text(json.dumps(clean(metadata), indent=2))
        target = data['img'].detach().float().cpu()
        recon = outputs['post_recon_combined'].detach().float().cpu()
        torch.save({'input': target, 'reconstruction': recon, 'data_idx': indices}, folder / 'batch.pt')
        for i in range(target.shape[0]):
            # Raw tensors above preserve NaNs and unclamped values for inspection.
            comparison = torch.cat([target[i], recon[i]])
            save_image(torch.nan_to_num(comparison).clamp(0, 1),
                       folder / f'sample-{i:02d}.png', nrow=target.shape[1])
        self.last_capture = m.it
        print(f'WARNING: {reason}: saved batch and diagnostics to {folder}', flush=True)
