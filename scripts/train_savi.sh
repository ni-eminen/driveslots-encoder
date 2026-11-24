#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo 'DATASET_ROOT=/path/to/obj3d-bev GPUS=4 bash scripts/train_savi.sh [extra training arguments]'
    echo 'Optional: OUTPUT_DIR=outputs/pretraining PARAMS=slotformer/bev_slots/configs/savi_bev_params.py'
    exit 0
fi
: "${DATASET_ROOT:?Set DATASET_ROOT to the directory containing train/ and val/}"
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-4}" \
    scripts/train.py --task bev_slots \
    --params "${PARAMS:-slotformer/bev_slots/configs/savi_bev_params.py}" \
    --dataset_path "$DATASET_ROOT" --output-dir "${OUTPUT_DIR:-outputs/pretraining}" --ddp "$@"
