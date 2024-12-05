import logging

import torch
from einops import rearrange, repeat
from torch import nn

from CameraControl.base.base import CameraControlLVDM
from CameraControl.motionctrl.motionctrl_modified_modules import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
    new_forward_for_unet,
)

mainlogger = logging.getLogger('mainlogger')


class MotionCtrl(CameraControlLVDM):
    def __init__(self,
                 pose_dim=12,
                 diffusion_model_trainable_param_list=[],
                 depth_predictor_config=None,
                 normalize_T0=False,
                 weight_decay=1e-2,
                 *args,
                 **kwargs):
        super(MotionCtrl, self).__init__(
            diffusion_model_trainable_param_list,
            False,  # pose_encoder_trainable
            None,   # pose_encoder_config
            depth_predictor_config,
            normalize_T0,
            weight_decay,
            *args,
            **kwargs,
        )

        bound_method = new_forward_for_unet.__get__(
            self.model.diffusion_model,
            self.model.diffusion_model.__class__
        )
        setattr(self.model.diffusion_model, 'forward', bound_method)

        for _name, _module in self.model.diffusion_model.named_modules():
            if _module.__class__.__name__ == 'TemporalTransformer':
                bound_method = new_forward_for_TemporalTransformer.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == 'TimestepEmbedSequential':
                bound_method = new_forward_for_TimestepEmbedSequential.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == 'BasicTransformerBlock':
                # SpatialTransformer only
                if _module.context_dim is None:  # BasicTransformerBlock of TemporalTransformer, only self attn, context_dim=None

                    bound_method = new_forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, 'forward', bound_method)

                    bound_method = new__forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

                    cc_projection = nn.Linear(_module.attn2.to_k.in_features + pose_dim, _module.attn2.to_k.in_features)
                    nn.init.zeros_(list(cc_projection.parameters())[0])
                    nn.init.eye_(list(cc_projection.parameters())[0][:_module.attn2.to_k.in_features, :_module.attn2.to_k.in_features])
                    nn.init.zeros_(list(cc_projection.parameters())[1])
                    cc_projection.requires_grad_(True)

                    _module.add_module('cc_projection', cc_projection)

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False,
                        return_cond_frame_index=False, return_cond_frame=False, return_original_input=False, rand_cond_frame=None,
                        enable_camera_condition=True, return_camera_data=False, return_video_path=False,
                        trace_scale_factor=1.0, cond_frame_index=None, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]

        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))

        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        if not hasattr(self, "null_prompt"):
            self.null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, self.null_prompt, cond_emb.detach())

        ## get conditioning frame
        if cond_frame_index is None:
            cond_frame_index = torch.zeros(batch_size, device=device, dtype=torch.long)
            rand_cond_frame = self.rand_cond_frame if rand_cond_frame is None else rand_cond_frame
            if rand_cond_frame:
                cond_frame_index = torch.randint(0, self.model.diffusion_model.temporal_length, (batch_size,), device=device)

        img = x[torch.arange(batch_size, device=device), :, cond_frame_index, ...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img)  ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[torch.arange(batch_size, device=device), :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond]  # b c t h w
            cond["c_cond_frame_index"] = cond_frame_index
            cond["origin_z_0"] = z.clone()
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)]  ## concat in the seq_len dim

        ########################################### only change here, add camera_condition input ###########################################
        if enable_camera_condition:
            with torch.no_grad():
                with torch.autocast('cuda', enabled=False):
                    w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
                    c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
                    B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

                    relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
                    relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

                    relative_w2c_RT_4x4 = relative_c2w_RT_4x4.inverse()

            cond["camera_condition"] = {
                "RT": rearrange(relative_w2c_RT_4x4[:,:,:3,:4], 'b t x y -> b t (x y)'),
            }

        ########################################### only change here, add camera_condition input ###########################################

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame_index:
            out.append(cond_frame_index)
        if return_cond_frame:
            out.append(x[torch.arange(batch_size, device=device), :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        if return_camera_data:
            camera_data = batch.get('camera_data', None)
            out.append(camera_data)

        if return_video_path:
            out.append(batch['video_path'])

        return out
