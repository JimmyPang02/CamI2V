import os
import random

import numpy as np
import omegaconf
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms


def make_spatial_transformations(resolution, type, ori_resolution=None):
    """
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms.RandomCropss(resolution)
    elif type == "resize_center_crop":
        is_square = (resolution[0] == resolution[1])
        if is_square:
            transformations = transforms.Compose([
                transforms.Resize(resolution[0], antialias=True),
                transforms.CenterCrop(resolution[0]),
            ])
        else:
            transformations = transforms.Compose([
                transforms.Resize(min(resolution)),
                transforms.CenterCrop(resolution),
            ])
    else:
        raise NotImplementedError
    return transformations


class RealEstate10K(Dataset):
    """
    RealEstate10K Dataset.
    For each video, its meta info is stored in a txt file whose contents are as follows:
    line 0: video_url
    line 1: empty
    line 2: caption

    In the rest, each line is a frame, including frame path, 4 camera intrinsics, and 3*4 camera pose (the matrix is row-major order).

    e.g.
    line 3: 0_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    line 4: 1_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    ...

    meta_path: path to the meta file
    meat_list: path to the meta list file
    data_dir: path to the data folder
    video_length: length of the video clip for training
    resolution: target resolution, a list of int, [h, w]
    frame_stride: stride between frames, int or list of int, [min, max], do not larger than 32 when video_length=16
    spatial_transform: spatial transformation, ["random_crop", "resize_center_crop"]
    count_globalsteps: whether to count global steps
    bs_per_gpu: batch size per gpu, used to count global steps

    """

    def __init__(self,
                 meta_path,
                 meta_list,
                 data_dir,
                 per_frame_scale_path=None,
                 camera_pose_sections=1,
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride=1,  # [min, max], do not larger than 32 when video_length=16
                 frame_stride_for_condition=0,
                 invert_video=False,
                 spatial_transform=None,
                 count_globalsteps=False,
                 bs_per_gpu=None,
                 RT_norm=False,
                 load_raw_resolution=True,
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride_for_condition = frame_stride_for_condition
        self.frame_stride = frame_stride
        self.spatial_transform_type = spatial_transform
        self.count_globalsteps = count_globalsteps
        self.bs_per_gpu = bs_per_gpu
        self.invert_video = invert_video
        self.RT_norm = RT_norm
        self.load_raw_resolution = load_raw_resolution
        self.camera_pose_sections = camera_pose_sections

        self.metadata = []
        with open(meta_list, 'r') as f:
            # self.metadata = [line.strip() for line in f.readlines()]
            self.metadata = np.array([line.strip() for line in f.readlines()], dtype=np.string_)
        
        if per_frame_scale_path:
            self.per_frame_scale = np.load(per_frame_scale_path, allow_pickle=True)['arr_0'].item()

        # make saptial transformations
        if isinstance(self.resolution[0], int):
            self.num_resolutions = 1
            self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
                if self.spatial_transform_type is not None else None
        else:
            # multiple resolutions training
            assert (isinstance(self.resolution[0], list) or isinstance(self.resolution[0], omegaconf.listconfig.ListConfig))
            self.num_resolutions = len(resolution)
            self.spatial_transform = None
            if self.num_resolutions > 1:
                assert (self.count_globalsteps)
        if self.count_globalsteps:
            assert (bs_per_gpu is not None)
            self.counter = 0

        print(f'============= length of dataset {len(self.metadata)} =============')

    def __getitem__(self, index):
        ## set up for dynamic resolution training
        if self.count_globalsteps:
            self.counter += 1
            self.global_step = self.counter // self.bs_per_gpu
        else:
            self.global_step = None

        to_inverse = (self.invert_video and random.random() > 0.5)

        ## get frames until success
        index = index % len(self.metadata)
        sample_name = self.metadata[index].decode('utf-8')
        with open(f"{self.meta_path}/{sample_name}.txt", 'r') as f:
            lines = f.readlines()
        caption = lines[2].strip()
        video_path = os.path.join(self.data_dir, lines[1].strip() + '.mp4')
        if self.load_raw_resolution:
            video_reader = VideoReader(video_path, ctx=cpu(0))
        else:
            video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
            assert len(video_reader) < self.video_length, print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")

        fps_ori = video_reader.get_avg_fps()
        lines = lines[3:]
        frame_num = len(lines)

        frame_stride_drop = 0
        while True:
            if isinstance(self.frame_stride, int):
                frame_stride = max(self.frame_stride - frame_stride_drop, 1)
            elif (isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig)) and len(self.frame_stride) == 2:  # [min, max]
                assert (self.frame_stride[0] <= self.frame_stride[1]), f"frame_stride[0]({self.frame_stride[0]}) > frame_stride[1]({self.frame_stride[1]})"
                frame_stride = random.randint(self.frame_stride[0], self.frame_stride[1])
            else:
                print(type(self.frame_stride))
                print(len(self.frame_stride))
                print(f"frame_stride={self.frame_stride}")
                raise NotImplementedError

            required_frame_num = frame_stride * (self.video_length - 1) + 1
            if frame_num < required_frame_num:
                if isinstance(self.frame_stride, int) and frame_num < required_frame_num * 0.5:
                    frame_stride_drop += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            break

            ## select a random clip
        random_range = frame_num - required_frame_num
        start_idx = random.randint(0, random_range) if random_range > 0 else 0
        frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
        # camera_pose_str, camera_pose, camera_intrinsics = "", [], []

        camera_data = torch.from_numpy(np.loadtxt(lines))[frame_indices].float() # [t, ]
        fx, fy, cx, cy = camera_data[:, 1:5].chunk(4, dim=-1) # [t,4]
        camera_pose_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
        camera_pose_4x4 = torch.cat([camera_pose_3x4, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * len(frame_indices))], dim=1)  # [t, 4, 4]

        frames = video_reader.get_batch(frame_indices)
        del video_reader

        ## process data
        assert (frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

        if self.num_resolutions > 1:
            ## make transformations based on the current resolution
            res_idx = self.global_step % 3
            res_curr = self.resolution[res_idx]
            self.spatial_transform = make_spatial_transformations(res_curr,
                                                                  self.spatial_transform_type,
                                                                  ori_resolution=frames.shape[2:])

        ## spatial transformations
        if self.spatial_transform is not None:
            if self.spatial_transform_type == 'resize_center_crop':
                sample_H, sample_W = self.resolution[0], self.resolution[1]
                if sample_H <= sample_W:
                    scale = sample_H / ori_H
                else:
                    scale = sample_W / ori_W
                fx *= ori_W * scale
                fy *= ori_H * scale
                cx *= sample_W
                cy *= sample_H

            frames = self.spatial_transform(frames)

        if self.resolution is not None:
            if self.num_resolutions > 1:
                assert (frames.shape[2] == res_curr[0] and frames.shape[3] == res_curr[1]), f'frames={frames.shape}, res_curr={res_curr}'
            else:
                assert (frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if to_inverse:
            # inverse frame order in dim=1
            frames = frames.flip(dims=(1,))

        _1, _0 = torch.ones_like(fx), torch.zeros_like(fx)
        camera_intrinsics = torch.hstack([fx, _0, cx, _0, fy, cy, _0, _0, _1]).reshape(-1, 3, 3) # [t, 3, 3]

        data = {
            'video': frames,    # [c,t,h,w]
            'caption': caption,
            'video_path': video_path,
            'fps': fps_clip,
            'frame_stride': frame_stride if self.frame_stride_for_condition == 0 else self.frame_stride_for_condition,
            'RT': camera_pose_4x4,  # Tx4x4
            'camera_data': camera_data,
            'camera_intrinsics': camera_intrinsics, # Tx3x3
            # 'trajs': torch.zeros(2, self.video_length, frames.shape[2], frames.shape[3])
        }

        if hasattr(self, "per_frame_scale"):
            data['per_frame_scale'] = torch.from_numpy(self.per_frame_scale[sample_name][frame_indices]).float()
        return data

    def __len__(self):
        return len(self.metadata)
