# DriveSlots encoder

Stochastic SAVi pretraining on RGB bird's-eye-view sequences for DriveSlots, based on SlotFormer. It learns object slots using reconstruction, perceptual, and KL losses. Supervised encoder fine-tuning and language-guided driving are in [driveslots](https://github.com/ni-eminen/driveslots).

## Model weights

The [pretrained SAVi encoder weights](https://huggingface.co/ni-eminen/savi-pretrained-30slots) are available on Hugging Face. Links to the fine-tuned encoder and instruction-finetuned models are in the [DriveSlots README](https://github.com/ni-eminen/driveslots#model-weights).

## Setup

Use a separate Python environment with a CUDA-compatible PyTorch/torchvision installation:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Training uses the upstream `nerv` v0.1.0 training framework. Dependencies include its original PyTorch Lightning API (`pytorch-lightning<2`). Authenticate with `wandb login` for online logging, or set `WANDB_MODE=offline`.

## Data

The dataset root contains `train/`, `val/`, and `test/`. Each split contains numbered sequence directories with consecutive JPEG frames (`0000.jpg`, `0001.jpg`, ...).

`scripts/data_preproc/create_obj3d_style_dataset.py` converts an extracted LMDrive dataset to this layout and writes a split manifest. **It moves the source birdview directories**, so run it on a disposable copy of the extracted data:

```bash
python scripts/data_preproc/create_obj3d_style_dataset.py --help
```

The split follows weather assignments in the code and reserves a fraction of the training-weather sequences for validation. Cropping and frame sampling happen in the dataset loader.

## Training

```bash
export DATASET_ROOT=/absolute/path/to/obj3d-bev
GPUS=4 OUTPUT_DIR=outputs/pretraining bash scripts/train_savi.sh
```

The model, losses, optimizer, batch sizes, and schedule are configured in `slotformer/bev_slots/configs/savi_bev_params.py`. Copy that configuration and set `PARAMS=/path/to/custom_params.py` for a separate experiment. Extra arguments are passed to `scripts/train.py`; `--weight /path/to/checkpoint.pth` resumes a checkpoint.

Checkpoints are written under `OUTPUT_DIR/<configuration-name>/models`. Reconstruction diagnostics are configured by `recon_save_root` and `loss_spike_root` in the parameter file. The default encoder uses 30 slots of width 256 and six-frame clips with a 192×192 crop.

## Visualization and evaluation

```bash
python scripts/visualize_savi_slots.py --help
python scripts/eval_object_metrics/eval_fg_ari_miou.py --help
python scripts/eval_object_metrics/eval_fg_reconstruction.py --help
```

The visualization tool renders full and individual slot reconstructions. The object evaluation tools use explicit checkpoints and dataset manifests; mask generation and manifest helpers are in the same directory.

## Attribution and license

Based on [SlotFormer](https://github.com/pairlab/SlotFormer) by Ziyi Wu and collaborators, with stochastic SAVi and BEV dataset adaptations for DriveSlots. Shared upstream model utilities are retained. See `LICENSE` (MIT) and the source copyright notices. Datasets and pretrained model weights are distributed separately.
