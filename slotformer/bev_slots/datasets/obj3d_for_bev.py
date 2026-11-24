import os
import sys
import os.path as osp
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from nerv.utils import glob_all, load_obj

from .utils import BaseTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Obj3DDatasetForBev(Dataset):
    """OBJ3D dataset from G-SWM."""

    def __init__(
        self,
        data_root,
        split,
        obj3d_transform,
        n_sample_frames=6,
        frame_offset=None,
        video_len=50,
    ):
        assert split in ['train', 'val', 'test']
        self.data_root = os.path.join(data_root, split)
        self.split = split
        self.obj3d_transform = obj3d_transform
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self.video_len = video_len

        # Get all numbers
        self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False
        #sys.stderr.write(f'SELFLEN after _get_sample_idx() with {self.__len__()} samples, ')
        #sys.stderr.flush()

    def _get_video_start_idx(self, idx):
        # if self.split == 'train':
            #sys.stderr.write("self.valid_idx\n")
            #sys.stderr.write(f"{len(self.valid_idx)}\n")
            #sys.stderr.write(f"-->{self.valid_idx}<--\n")
            #sys.stderr.flush()
        try:
            a = self.valid_idx[idx]
            return a
        except Exception as e:
            sys.stderr.write(f"Error in _get_video_start_idx with idx {idx} and valid_idx length {len(self.valid_idx)}\n")
            sys.stderr.flush()
            raise e

    def _read_frames(self, idx):
        folder, start_idx = self._get_video_start_idx(idx)
        start_idx += 0  # files start from '0000.jpg'
        filename = osp.join(folder, '{:04d}.jpg')
        frames = [
            Image.open(filename.format(start_idx +
                                       n * self.frame_offset)).convert('RGB')
            for n in range(self.n_sample_frames)
        ]
        frames = [self.obj3d_transform(img) for img in frames]
        return torch.stack(frames, dim=0)  # [N, C, H, W]

    def _read_bboxes(self, idx):
        """Load empty bbox and pres mask for compatibility."""
        bboxes = np.zeros((self.n_sample_frames, 5, 4))
        pres_mask = np.zeros((self.n_sample_frames, 5))
        return bboxes, pres_mask

    def get_video(self, video_idx):
        folder = self.files[video_idx]
        num_frames = (self.video_len + 1) // self.frame_offset
        filename = osp.join(folder, 'test_{}.png')
        frames = [
            Image.open(filename.format(1 +
                                       n * self.frame_offset)).convert('RGB')
            for n in range(num_frames)
        ]
        frames = [self.obj3d_transform(img) for img in frames]
        return {
            'video': torch.stack(frames, dim=0),
            'data_idx': video_idx,
        }

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - bbox: [T, max_num_obj, 4], empty, for compatibility
            - pres_mask: [T, max_num_obj], empty, for compatibility
        """
        if self.load_video:
            return self.get_video(idx)

        #sys.stderr.write(f'SELFLEN on getitem {self.__len__()}')
        #sys.stderr.flush()

        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'img': frames,
        }
        if self.split != 'train':
            bboxes, pres_mask = self._read_bboxes(idx)
            data_dict['bbox'] = torch.from_numpy(bboxes).float()
            data_dict['pres_mask'] = torch.from_numpy(pres_mask).bool()
        return data_dict

    def _get_sample_idx(self):
        valid_idx = []  # (video_folder, start_idx)
        files = glob_all(self.data_root, only_dir=True)
        self.files = [s.rstrip('/') for s in files]
        self.num_videos = len(self.files)
        for folder in self.files:
            video_len = len(glob_all(folder))
            # simply use random uniform sampling
            if self.split == 'train':
                max_start_idx = video_len - \
                    (self.n_sample_frames - 1) * self.frame_offset
                valid_idx += [(folder, idx) for idx in range(max_start_idx)]
            # only test once per video
            else:
                valid_idx += [(folder, 0)]
        return valid_idx

    def __len__(self):
        if self.load_video:
            return len(self.files)
        return len(self.valid_idx)


class Obj3DSlotsDataset(Obj3DDatasetForBev):
    """OBJ3D dataset from G-SWM with pre-computed slots."""

    def __init__(
        self,
        data_root,
        video_slots,
        split,
        obj3d_transform,
        n_sample_frames=16,
        frame_offset=None,
        video_len=50,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            obj3d_transform=obj3d_transform,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            video_len=video_len,
        )

        # pre-computed slots
        self.video_slots = video_slots

    def _read_slots(self, idx):
        """Read video frames slots."""
        folder, start_idx = self.valid_idx[idx]
        slots = self.video_slots[os.path.basename(folder)]  # [T, N, C]
        slots = [
            slots[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(slots, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - bbox: [T, max_num_obj, 4], empty, for compatibility
            - pres_mask: [T, max_num_obj], empty, for compatibility
            - slots: [T, N, C] slots extracted from OBJ3D video frames
        """
        slots = self._read_slots(idx)
        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'slots': slots,
            'img': frames,
        }
        if self.split != 'train':
            bboxes, pres_mask = self._read_bboxes(idx)
            data_dict['bbox'] = torch.from_numpy(bboxes).float()
            data_dict['pres_mask'] = torch.from_numpy(pres_mask).bool()
        return data_dict


def build_obj3d_for_bev_dataset(params, val_only=False):
    """Build OBJ3D video dataset."""
    if params.crop_bevs:
        transform = transforms.Compose([
            transforms.CenterCrop((192, 192)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor()
        ])


    args = dict(
        data_root=params.data_root,
        split='val',
        obj3d_transform=transform,
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
    )
    val_dataset = Obj3DDatasetForBev(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = Obj3DDatasetForBev(**args)
    return train_dataset, val_dataset


def build_obj3d_for_bev_slots_dataset(params, val_only=False):
    """Build OBJ3D video dataset with pre-computed slots."""
    if params.crop_bevs:
        transform = transforms.Compose([
            transforms.CenterCrop((192, 192)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor()
        ])

    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=slots['val'],
        split='val',
        obj3d_transform=transform,
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
    )
    val_dataset = Obj3DSlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = Obj3DSlotsDataset(**args)
    return train_dataset, val_dataset
