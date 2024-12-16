import json
import os
from uuid import uuid4

import numpy as np
import open3d as o3d
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import Tensor

from CameraControl.cameractrl.cameractrl import CameraCtrl
from CameraControl.CamI2V.cami2v import CamI2V
from CameraControl.data.single_image_for_inference import SingleImageForInference
from CameraControl.data.utils import camera_pose_lerp, create_line_point_cloud, relative_pose
from CameraControl.motionctrl.motionctrl import MotionCtrl
from utils.utils import instantiate_from_config


def default(a, b):
    return a if a is not None else b


def bezier_curve(t: Tensor, a: float, b: float):
    points = torch.tensor([[0.0, 0.0], [default(a, 0.5), 0.0], [default(b, 0.5), 1.0], [1.0, 1.0]], dtype=t.dtype)
    coeffs = torch.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3])

    return points.T @ coeffs  # [2, num_frames]


def camera_pose_lerp_bezier(c2w: Tensor, target_frames: int, coef_a: float, coef_b: float):
    t = torch.linspace(0, 1, target_frames, dtype=c2w.dtype)
    xs, ys = bezier_curve(t, coef_a, coef_b).contiguous()

    right_indices = torch.searchsorted(xs, t)
    left_indices = (right_indices - 1).clamp(0)

    x_weights = ((t - xs[left_indices]) / (xs[right_indices] - xs[left_indices]).clamp(1e-9)).clamp(0.0, 1.0)
    y_weights = torch.lerp(ys[left_indices], ys[right_indices], x_weights) * (c2w.shape[0] - 1)

    left_indices = y_weights.floor().long()
    right_indices = y_weights.ceil().long()

    return torch.lerp(c2w[left_indices], c2w[right_indices], y_weights.unsqueeze(-1).unsqueeze(-1).frac())


class Image2Video:
    def __init__(
        self,
        result_dir: str = "./gradio_results",
        model_meta_file: str = "./models.json",
        video_length: int = 16,
        save_fps: int = 10,
        device: str = "cuda",
    ):
        self.result_dir = result_dir
        self.model_meta_file = model_meta_file
        self.video_length = video_length
        self.save_fps = save_fps
        self.device = device

        os.makedirs(self.result_dir, exist_ok=True)

        self.models: dict[str, MotionCtrl | CameraCtrl | CamI2V] = {}
        self.single_image_processors: dict[str, SingleImageForInference] = {}

    def load_model(self, config_file: str, ckpt_path: str, width: int, height: int):
        config = OmegaConf.load(config_file)
        config.model.params.perframe_ae = True
        model: MotionCtrl | CameraCtrl | CamI2V = instantiate_from_config(config.model).to(self.device).eval()

        if model.rescale_betas_zero_snr:
            model.register_schedule(
                given_betas=model.given_betas,
                beta_schedule=model.beta_schedule,
                timesteps=model.timesteps,
                linear_start=model.linear_start,
                linear_end=model.linear_end,
                cosine_s=model.cosine_s,
            )
        model = model.to(self.device)

        model.eval()
        for n, p in model.named_parameters():
            p.requires_grad = False

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "module" in state_dict:  # deepspeed checkpoint
                state_dict = state_dict["module"]
            elif "state_dict" in state_dict:  # lightning checkpoint
                state_dict = state_dict["state_dict"]
            state_dict = {k.replace("framestride_embed", "fps_embedding"): v for k, v in state_dict.items()}
            try:
                model.load_state_dict(state_dict)
                print("successfully loaded checkpoint {}".format(ckpt_path))
            except Exception as e:
                print(e)
                model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        # print("model dtype", model.dtype)

        single_image_processor = SingleImageForInference(
            video_length=self.video_length,
            resolution=(height, width),
            spatial_transform_type="resize_center_crop",
            device=self.device,
        )

        return model, single_image_processor

    def save_pcd(self, name: str, points: np.ndarray, colors: np.ndarray) -> tuple[str, str]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        o3d.io.write_point_cloud(f"{self.result_dir}/{name}.ply", pcd)
        mesh = o3d.io.read_triangle_mesh(f"{self.result_dir}/{name}.ply")
        o3d.io.write_triangle_mesh(f"{self.result_dir}/{name}.obj", mesh)

        return f"{self.result_dir}/{name}.obj"

    @torch.no_grad
    @torch.autocast("cuda", enabled=True)
    def get_image(
        self,
        model_name: str,
        ref_img: Image.Image,
        caption: str,
        camera_pose_type: str,
        trace_extract_ratio: float = 1.0,
        frame_stride: int = 1,
        steps: int = 25,
        trace_scale_factor: float = 1.0,
        camera_cfg: float = 1.0,
        cfg_scale: float = 7.5,
        seed: int = 123,
        enable_camera_condition: bool = True,
        use_bezier_curve: bool = False,
        bezier_coef_a: float = None,
        bezier_coef_b: float = None,
        loop: bool = False,
        cond_frame_index: int = 0,
        eta: float = 1.0,
        ref_img2: Image.Image = None,
    ):
        with open("prompts/camera_pose_files/meta_data.json", "r", encoding="utf-8") as f:
            camera_pose_file_path = json.load(f)[camera_pose_type]

        camera_data = torch.from_numpy(np.loadtxt(camera_pose_file_path, comments="https"))  # t, -1
        w2cs_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
        w2cs_4x4 = torch.cat(
            [w2cs_3x4, torch.tensor([[[0, 0, 0, 1]]] * w2cs_3x4.shape[0], device=w2cs_3x4.device)], dim=1
        )  # [t, 4, 4]
        c2ws_4x4 = w2cs_4x4.inverse()[: max(1, int(w2cs_4x4.shape[0] * trace_extract_ratio))]  # [t, 4, 4]
        if use_bezier_curve:
            c2ws_4x4 = camera_pose_lerp_bezier(c2ws_4x4, c2ws_4x4.shape[0], bezier_coef_a, bezier_coef_b)
        if loop:
            c2ws_4x4 = torch.cat([c2ws_4x4, c2ws_4x4.flip(0)], dim=0)
        c2ws_lerp_4x4 = camera_pose_lerp(c2ws_4x4, self.video_length)  # [video_length, 4, 4]
        w2cs_lerp_4x4 = c2ws_lerp_4x4.inverse()  # [video_length, 4, 4]
        rel_c2ws_lerp_4x4 = relative_pose(c2ws_lerp_4x4, mode="left", ref_index=cond_frame_index).clone()
        rel_c2ws_lerp_4x4[:, :3, 3] = rel_c2ws_lerp_4x4[:, :3, 3] * trace_scale_factor

        for k, v in filter(lambda x: x[0] != model_name, self.models.items()):
            self.models[k] = v.cpu()

        if model_name not in self.models:
            with open(self.model_meta_file, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)[model_name]
            print(f"loading model {model_name}, metadata:", model_metadata)
            model, single_image_preprocessor = self.load_model(**model_metadata)

            self.models[model_name] = model
            self.single_image_processors[model_name] = single_image_preprocessor
            print("models loaded:", list(self.models.keys()))

        model = self.models[model_name].to(self.device)
        single_image_preprocessor = self.single_image_processors[model_name]
        print("using", model_name)

        torch.cuda.empty_cache()

        seed_everything(seed)
        log_images_kwargs = {
            "ddim_steps": steps,
            "ddim_eta": eta,
            "unconditional_guidance_scale": cfg_scale,
            "timestep_spacing": "uniform_trailing",
            "guidance_rescale": 0.7,
            "camera_cfg": camera_cfg,
            "camera_cfg_scheduler": "constant",
            "enable_camera_condition": enable_camera_condition,
            "trace_scale_factor": trace_scale_factor,
            "result_dir": self.result_dir,
        }

        frame_indices = list(range(0, self.video_length))

        input = single_image_preprocessor.get_batch_input(
            ref_img, caption, w2cs_lerp_4x4[frame_indices, :3], frame_stride, ref_img2=ref_img2
        )
        input["cond_frame_index"] = torch.tensor(
            [cond_frame_index] * input["video"].shape[0], device=input["video"].device, dtype=torch.long
        )
        log_images_kwargs["cond_frame_index"] = input["cond_frame_index"].clone()

        output = model.log_images(input, **log_images_kwargs)
        video_clip = output["samples"].clamp(-1.0, 1.0).cpu()

        full_video = video_clip.clone()  # b, c, f, h, w

        points, colors = [], []
        for frame_idx, rel_c2w in enumerate(rel_c2ws_lerp_4x4[frame_indices, :3]):
            right, up, forward, camera_center = rel_c2w.unbind(-1)
            start_point = camera_center
            end_point = camera_center + forward * 0.2

            camera, camera_colors = create_line_point_cloud(
                start_point=start_point,
                end_point=end_point,
                num_points=200,
                color=np.array([0, 1.0, 0]),
            )

            points.append(camera)
            colors.append(camera_colors)

        scene_with_camera_path = self.save_pcd(
            "output_with_cam", np.concatenate(points, axis=0), np.concatenate(colors, axis=0)
        )

        print(full_video.shape)
        video_path = f"{self.result_dir}/{model_name}_{uuid4().fields[0]:x}.mp4"
        self.save_video(full_video.clamp(-1.0, 1.0), video_path)

        return_list = [scene_with_camera_path, video_path]

        return return_list

    def save_video(self, video: Tensor, path: str):
        n, c, t, h, w = video.shape
        # print(video.shape)
        video = torch.nn.functional.interpolate(
            rearrange(video, "n c t h w -> (n t) c h w"), (h // 2 * 2, w // 2 * 2), mode="bilinear"
        )
        video = rearrange(video, "(n t) c h w -> t n c h w", n=n, t=t)
        # print(video.shape)
        # video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(1), padding=0) for framesheet in video
        ]  # [3, n*h, 1*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.io.write_video(path, grid, fps=self.save_fps, video_codec="h264", options={"crf": "10"})

        return path


if __name__ == "__main__":
    i2v = Image2Video()
    video_path = i2v.get_image("prompts/art.png", "man fishing in a boat at sunset")
    print("done", video_path)
