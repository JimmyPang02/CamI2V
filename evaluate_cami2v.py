"""
评估文件应该分成三部分：
1. 加载RealEstate10K数据集
2. 把数据集转成对应测试模型(如motionctrl)的输入格式，并完成推理测试
3. 根据推理测试结果，计算评估指标
"""
import os
import glob
import numpy as np
import random
import torch
import torchvision
from torchvision.io import read_video
import json
from main.inference.motionctrl_cmcm_evaluate import run_motionctrl_inference
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

import sys
sys.path.append('../FVD')
# from frechet_video_distance import frechet_video_distance as fvd

"""
1. 加载RealEstate10K数据集
"""
def parse_trajectory_file(txt_path, device=torch.device('cuda'),num_frames=25):
    """
    针对RealEstate10K的相机轨迹
    解析单个相机轨迹 .txt 文件，仅获取相机外参矩阵。
    
    参数:
        txt_path (str): 相机轨迹文件路径。
        device (torch.device): 张量设备（'cpu' 或 'cuda'）。
    
    返回:
        video_url (str): 视频URL/ID，用于获取原始视频的路径。
        c2ws (torch.Tensor): 相机外参矩阵 [N, 3, 4] 或标志位 'NOT_ENOUGH_FRAMES'
    """
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 第1行是视频URL
    video_url = lines[0]

    # 解析相机外参
    w2cs = []
    for frame_line in lines[1:]:
        cols = frame_line.split()
        if len(cols) >= 19:  # 基础检查：确保至少有19列
            ext_cols = list(map(float, cols[7:19]))  # 提取外参（7-19列）
            ext_matrix = np.array(ext_cols).reshape(3, 4)
            w2cs.append(torch.tensor(ext_matrix, dtype=torch.float).to(device))

    # 转换为Tensor并计算c2w
    w2cs = torch.stack(w2cs, dim=0)  # [N, 3, 4]
    bottom = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float).to(device)
    bottom = bottom.repeat(w2cs.shape[0], 1, 1)  # [N, 1, 4]
    w2cs_4x4 = torch.cat((w2cs, bottom), dim=1)  # [N, 4, 4]
    c2ws_4x4 = torch.linalg.inv(w2cs_4x4)        # [N, 4, 4]
    c2ws = c2ws_4x4[:, :3, :]                   # [N, 3, 4]
    
    # 均匀采样num_frames帧
    if len(c2ws) < num_frames:
        return video_url, 'NOT_ENOUGH_FRAMES'
    
    indices = np.linspace(0, len(c2ws) - 1, num_frames, dtype=int)
    c2ws = c2ws[indices]
    print(f"c2ws shape: {c2ws.shape}")
    
    return video_url, c2ws

"""
1. 加载RealEstate10K数据集
"""
def sample_frames(frame_folder, num_frames=25):
    """
    针对RealEstate10K
    从帧文件夹中均匀或随机采样帧图像。
    
    参数:
        frame_folder (str): 帧文件夹路径。
        num_frames (int): 采样的帧数。
        
    返回:
        frame_paths (list[str]): 采样的帧路径。
    """
    if not os.path.exists(frame_folder):
        return []
    
    # 获取所有帧文件
    all_frames = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
    if len(all_frames) == 0:
        return []
    
    # 如果帧数不足num_frames，直接返回所有帧
    if len(all_frames) <= num_frames:
        return all_frames
    
    # 均匀采样num_frames帧
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    sampled_frames = [all_frames[i] for i in indices]
    return sampled_frames

"""
1. 加载RealEstate10K数据集
"""
def load_RealEstate10K(
    camera_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/RealEstate10K_camera/test",
    dataset_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/RealEstate10K/dataset/test",
    video_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/RealEstate10K/videos/",
    num_frames=14,
    sample_num=1000
):
    """
    1. 遍历 RealEstate10K_camera/test 下所有 .txt 文件，采样sample_num个视频
    2. 解析相机轨迹，获取 extrinsics
    3. 从 dataset/test 下对应帧文件夹采样 num_frames 帧
    4. 定位到视频文件
    """
    # 获取所有轨迹文件
    txt_files = glob.glob(os.path.join(camera_root, "*.txt"))
    
    results = []  # 存放解析结果
    
    cnt=0
    while len(results) < sample_num: # 循环sample_num个视频
        print(f"cnt:{cnt}")
        print(f"len(results):{len(results)}")
        print(len(results))
        print(f"{len(results)}/{sample_num}")
        print(f"{cnt}/{len(txt_files)}")
        
        # 获取轨迹文件路径               
        txt_path = txt_files[cnt]
        cnt=cnt+1

        # 获取视频ID（基于 txt 文件名）
        basename = os.path.splitext(os.path.basename(txt_path))[0]  # e.g., "2bec33eeeab0bb9d"
        
        # debug
        if cnt > 10:
            print("break")
            break
        
        # 如果所有文件都检查完了，退出循环
        if cnt >= len(txt_files):
            break  
        
        # 检查对应的帧文件夹是否存在
        frame_folder = os.path.join(dataset_root, basename)
        if not os.path.exists(frame_folder):
            print(f"No frame folder found for {txt_path}")
            continue  # 如果帧文件夹不存在，跳过该轨迹文件
        
        # 检查帧文件夹中是否有图片
        frame_paths = sample_frames(frame_folder, num_frames=num_frames)
        if not frame_paths:
            print(f"No frames found for {txt_path}")
            continue  # 如果帧文件夹中没有图片，跳过该轨迹文件

        # 解析轨迹
        video_url, extrinsics = parse_trajectory_file(txt_path,device=torch.device('cuda'), num_frames=num_frames)
        # 如果外参矩阵不足sample_num个，跳过该视频
        if extrinsics == 'NOT_ENOUGH_FRAMES':
            print(f"Not enough extrinsics for {txt_path}")
            continue

        # 原始视频路径
        video_file = os.path.join(video_root, f"{basename}.mp4")
        if not os.path.exists(video_file):
            video_file = None

        # 组装结果
        info_dict = {
            "txt_file": txt_path,
            "video_id": basename,
            "video_url": video_url,
            "video_file": video_file,
            "extrinsics": extrinsics,  # [N, 3, 4]
            "frame_paths": frame_paths  # 采样的num_frames帧图像路径
        }

        results.append(info_dict)

    return results

"""
1. 加载RealEstate10K数据集
"""
def load_real_frames(frame_paths, height=576, width=1024):
    """
    从帧路径列表中加载帧图像，并调整大小。
    
    参数:
        frame_paths (list[str]): 帧路径列表。
        height (int): 目标高度。
        width (int): 目标宽度。
        
    返回:
        real_frames (torch.Tensor): 调整大小后的帧图像张量 [N, C, H, W]。
    """
    real_frames = []
    for frame_path in frame_paths:
        frame = torchvision.io.read_image(frame_path).float() / 255.0
        frame = torchvision.transforms.Resize((height, width))(frame)
        real_frames.append(frame)
    real_frames = torch.stack(real_frames, dim=0)
    return real_frames


"""
2. 把数据集转成对应测试模型(如motionctrl)的输入格式，并完成推理测试
"""
def save_extrinsics_to_json(parsed_results, output_json_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/MotionCtrl/examples/camera_poses_evaluate"):
    """
    将从 load_RealEstate10K() 获取的相机外参（extrinsics）保存到指定 JSON 文件夹中
    将RealEstate10K 轨迹格式转为motionctrl的轨迹格式
    
    参数:
        parsed_results (list): 从 load_RealEstate10K() 获取的解析结果列表。
        output_json_root (str): 保存 JSON 文件的根目录。
    """
    os.makedirs(output_json_root, exist_ok=True)

    for item in parsed_results:
        video_id = item["video_id"]
        extrinsics = item["extrinsics"]  # [N, 3, 4]
        json_path = os.path.join(output_json_root, f"{video_id}_pose.json")

        # 转换为 [N, 12] 格式并保存
        extrinsics_flat = extrinsics.reshape(-1, 12).cpu().tolist()
        with open(json_path, 'w') as f:
            json.dump(extrinsics_flat, f, indent=4)
        
        print(f"Saved extrinsics for {video_id} to {json_path}")
    


"""
2. 把数据集转成对应测试模型(如motionctrl)的输入格式，并完成推理测试
"""
def run_motionctrl(RealEstate10K_parsed_results):
    # motionctrl 相机外参保存路径
    output_pose_dir="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/MotionCtrl/examples/camera_poses_evaluate"
    # 保存相机外参到 JSON 文件(RealEstate轨迹转成motionctrl格式)
    save_extrinsics_to_json(RealEstate10K_parsed_results, output_json_root=output_pose_dir)

    # 打印前几个结果进行检查
    for item in parsed_results[:3]:
        print("轨迹文件:", item["txt_file"])
        print("视频ID:", item["video_id"])
        print("视频URL:", item["video_url"])
        print("原始视频文件:", item["video_file"])
        print("外参矩阵形状:", item["extrinsics"].shape)
        print("采样的帧图像路径 (前5帧):", item["frame_paths"][:5])
        # print(len(item["frame_paths"]))
        print("-------------------------------------------------\n")


    # 初始化 FID 计算器
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=False, input_img_size=(3, 299, 299), feature_extractor_weights_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/model/FID/weights-inception-2015-12-05-6726825d.pth")
    fid_values = [ ]
        
    # 遍历 parsed_results，调用 run_motionctrl_inference（运行motionctrl测试）
    for idx, item in enumerate(RealEstate10K_parsed_results):
        video_id = item["video_id"]
        frame_paths = item["frame_paths"]
        extrinsics = item["extrinsics"]

        if not frame_paths:
            print(f"[{idx}] {video_id} has no frames, skip.")
            continue
        
        # 取第 1 帧作为 input
        image_input = frame_paths[0]
        
        # 根据video_id 获取对应的 .json 文件
        pose_path = os.path.join(output_pose_dir, f"{video_id}_pose.json")
        print(f"Pose path: {pose_path}")
        
        # 读取，查看shape
        import cv2
        image = cv2.imread(image_input)
        print(f"Image shape: {image.shape}")

        print(f"\n=== [{idx}] Start inference for video_id={video_id} ===")
        print(f"Use first frame: {image_input}")

        # 调用推理函数
        generated_videos, generated_frames = run_motionctrl_inference(
            seed=12345,
            ckpt="../../model/motionctrl/motionctrl_svd.ckpt",
            config="configs/inference/config_motionctrl_cmcm.yaml",
            savedir=f"outputs/motionctrl_svd/{video_id}",  # 每个视频ID单独输出
            savefps=10,
            ddim_steps=25,
            frames=14, 
            image_input=image_input,
            fps=10,
            motion=127,
            cond_aug=0.02,
            decoding_t=1,
            resize=True,
            height=576,
            width=1024,
            sample_num=1,
            transform=True,
            pose_dir=pose_path,
            speed=2.0,
            save_images=True,
            device="cuda"
        )
        """
        3. 根据推理测试结果，计算评估指标
        在线计算的指标可以在这里算，如torchmetric的FID
        """
        # generated_videos 原视频文件路径(基本全是None，感觉轨迹对应的id和video名称对不上)
        # generated_frames[0] 生成视频帧 torch.Size([14, 576, 1024, 3])
        
        # 加载生成帧
        generated_frames=generated_frames[0].permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        # 加载真实帧
        real_frames = load_real_frames(frame_paths, height=576, width=1024)

        # 计算FID
        fid_value = calculate_fid(generated_frames, real_frames,fid)
        fid_values.append(fid_value)

    print(f"FID value: {fid_values}")


def calculate_fid(generated_frames, real_frames, fid):
    """
    计算生成图像和真实图像之间的 FID 值。

    参数:
        generated_frames (torch.Tensor): 生成的图像张量，形状为 [N, C, H, W]。
        real_frames (torch.Tensor): 真实的图像张量，形状为 [N, C, H, W]。
        fid (FrechetInceptionDistance): 已经初始化的 FID 计算器。
        N代表N张图

    返回:
        fid_value (float): FID 值。
    """
    print("-------------calculate_fid--------------")
    print(generated_frames.shape)
    print(real_frames.shape)
    # 定义调整图像尺寸的 transform
    resize_transform = transforms.Compose([
        # transforms.Resize((299, 299)),  # 调整图像尺寸为 299x299
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    
    # 确保输入数据为 uint8 类型
    if generated_frames.dtype != torch.uint8:
        generated_frames = generated_frames.to(torch.uint8)
    if real_frames.dtype != torch.uint8:
        real_frames = real_frames.to(torch.uint8)

    # 调整生成图像的尺寸
    generated_frames_resized = torch.stack([resize_transform(frame) for frame in generated_frames])

    # 调整真实图像的尺寸
    real_frames_resized = torch.stack([resize_transform(frame) for frame in real_frames])

    # 更新 FID 计算器
    fid.update(real_frames_resized, real=True)
    fid.update(generated_frames_resized, real=False)

    # 计算 FID
    fid_value = fid.compute()
    print(f"FID: {fid_value}")
    fid.reset()  # 重置 FID 计算器，以便下次使用
    
    return fid_value
     
"""
评估文件应该分成三部分：
1. 加载RealEstate10K数据集
2. 把数据集转成对应测试模型(如motionctrl)的输入格式，并完成推理测试
3. 根据推理测试结果，计算评估指标
"""
if __name__ == "__main__":
     
    # 获取测试数据集信息
    parsed_results = load_RealEstate10K()
    print("RealEstate10K loaded")
    
    # 在对比模型上跑测试数据集，保存推理结果
    run_motionctrl(RealEstate10K_parsed_results=parsed_results)

    # 计算测评指标
    # 分为在线计算(torchmetric的FID可以不断update)和离线计算(得等结果都跑完保存成文件后，才能读取计算)
    # a. 在线计算 应该写在测试算法的run程序里
    # b. 离线计算 没必要写在这，独立开来写就行
