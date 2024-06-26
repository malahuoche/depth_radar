# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import torch
import torchvision
import numpy as np
from scipy.ndimage import rotate
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf
import cv2
from ..models.utils import deg2rad, rotmat2d
from ..osm.tiling import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera
from .image import pad_image, rectify_image, resize_image
from .utils import decompose_rotmat, random_flip, random_rot90,load_lidar
from PIL import Image
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('/home/classlab2/root/OrienterNet')
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToPILImage
import math
from PIL import Image, ImageStat
from .data_utils import load_depth,load_image
import random
class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
        "total_points_sampled":4,
        "sample_probability_lidar": 0.10,
        "patch_size": 512,#self.patch_size = patch_size
        "pad_size_x": 32 ,#self.pad_size_x = patch_size[1] // 2(288/2)
        "data_format":'CHW'
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, seq, name = self.names[idx]
        if self.cfg.init_from_gps:
            # print("Keys available in self.data:", self.data.keys())
            latlon_gps = self.data["gps_position"][idx][-2:].clone().numpy()
            # print(latlon_gps)
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        else:
            xy_w_init = self.data["t_c2w"][idx][:2].clone().double().numpy()

        if "shifts" in self.data:
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            R_c2w = rotmat2d((90 - yaw) / 180 * np.pi).float()
            error = (R_c2w @ self.data["shifts"][idx][:2]).numpy()
        else:
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)
        xy_w_init += error * self.cfg.max_init_error

        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        return self.get_view(idx, scene, seq, name, seed, bbox_tile)

    def get_view(self, idx, scene, seq, name, seed, bbox_tile):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()

        if "roll_pitch_yaw" in self.data:
            roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()
        else:
            roll, pitch, yaw = decompose_rotmat(self.data["R_c2w"][idx].numpy())
        # image = read_image(self.image_dirs[scene] / (name + self.image_ext))


        #加上雷达
        image_path = str(self.image_dirs[scene] / (name + self.image_ext))
        camera_image_path = image_path.replace("/radar/cart", "/camera_512")
        
        N_camera_image_path = image_path.replace("/radar/cart", "/camera_512")
        #oxford
        # camera_image_path = image_path.replace("/cart", "/image_new_r2")
        camera_image = read_image(Path(camera_image_path))
        padding = ((0, 0), (0, 0), (self.cfg.pad_size_x, self.cfg.pad_size_x))
        N_image = load_image(
            N_camera_image_path,
            normalize=False,
            data_format=self.cfg.data_format)

        N_image = np.pad(
            N_image,
            pad_width= padding,
            mode='edge')

        # print(camera_image_path)
        #读雷达的图像
        radar_image_path = image_path
        radar_image_path = radar_image_path.replace("cart", "cart_ro")

        # radar_image_path = radar_image_path.replace("Navtech_Cartesian", "Navtech_Cartesian_r")
        #boreas读点云图像
        # radar_image_path = image_path.replace("/radar/cart", "/lidar_cart")
        #oford 读点云
        # radar_image_path = image_path.replace("/cart", "/lidar_cart")
        # # print(radar_image_path)
        # radar_image = read_image(Path(radar_image_path))

        #depth---
        #读取雷达点云
        radar_points_path = radar_image_path.replace("cart_ro", "point_clouds_projected")
        radar_points_path = radar_points_path.replace(".png", ".bin")
        radar_points = load_lidar(radar_points_path)
        # print(radar_points.shape[0])
        #读取groun_truth
        # ground_truth_paths = radar_image_path.replace("cart_ro", "ground_truth")
        # if radar_points.ndim == 1:
        #     # Expand to 1 x 3
        #     radar_points = np.expand_dims(radar_points, axis=0)
        bounding_boxes_list = []
        # if radar_points.shape[0] <= self.cfg.total_points_sampled:
        #     radar_points = np.repeat(radar_points, 100, axis=0)
        # random_idx = np.random.randint(radar_points.shape[0], size=self.cfg.total_points_sampled)
        # radar_points = radar_points[random_idx,:]
        # Load ground truth depth
        # ground_truth = load_depth(
        #     ground_truth_paths,
        #     data_format=self.cfg.data_format)
        # if random.random() < self.cfg.sample_probability_of_lidar:

        #     ground_truth_for_sampling = np.copy(ground_truth)
        #     ground_truth_for_sampling = ground_truth_for_sampling.squeeze()
        #     # Find lidar points with depth greater than 1
        #     idx_lidar_samples = np.where(ground_truth_for_sampling > 1)

        #     # randomly sample total_points_sampled number of points from the lidar
        #     random_indices = random.sample(range(0, len(idx_lidar_samples[0])), self.total_points_sampled)

        #     points_x = idx_lidar_samples[1][random_indices]
        #     points_y = idx_lidar_samples[0][random_indices]
        #     points_z = ground_truth_for_sampling[points_y, points_x]

        #     noise_for_fake_radar_x = np.random.normal(0,25,radar_points.shape[0])
        #     noise_for_fake_radar_z = np.random.uniform(low=0.0, high=0.4, size=radar_points.shape[0])

        #     fake_radar_points = np.copy(radar_points)
        #     fake_radar_points[:,0] = points_x + noise_for_fake_radar_x
        #     fake_radar_points[:,0] = np.clip(fake_radar_points[:,0], 0, ground_truth_for_sampling.shape[1])
        #     fake_radar_points[:,2] = points_z + noise_for_fake_radar_z
        #     # we keep the y as the same it is since it is erroneous

        #     # convert x and y indices back to int after adding noise
        #     fake_radar_points[:,0] = fake_radar_points[:,0].astype(int)
        #     fake_radar_points[:,1] = fake_radar_points[:,1].astype(int)

        #     radar_points = np.copy(fake_radar_points)


        # get the shifted radar points after padding
        for radar_point_idx in range(0,radar_points.shape[0]):
            # Set radar point to the center of the patch
            radar_points[radar_point_idx,0] = radar_points[radar_point_idx,0] + self.cfg.pad_size_x

            bounding_box = [0,0,0,0]
            bounding_box[0] = radar_points[radar_point_idx,0] - self.cfg.pad_size_x
            bounding_box[1] = 0
            bounding_box[2] = radar_points[radar_point_idx,0] + self.cfg.pad_size_x
            bounding_box[3] = self.cfg.patch_size
            bounding_boxes_list.append(np.asarray(bounding_box))

        # ground_truth = np.pad(
        #     ground_truth,
        #     pad_width=self.padding,
        #     mode='constant',
        #     constant_values=0)

        # ground_truth_crops = []
        # Crop image and ground truth
        for radar_point_idx in range(0,radar_points.shape[0]):
            start_x = int(radar_points[radar_point_idx,0] - self.cfg.pad_size_x)
            end_x = int(radar_points[radar_point_idx,0] + self.cfg.pad_size_x)
            start_y = N_image.shape[-2] - self.cfg.patch_size

            # ground_truth_cropped = ground_truth[:, start_y:, start_x:end_x]
            # ground_truth_crops.append(ground_truth_cropped)
       
        # N_image = N_image[:, start_y:, ...]
        
        # ground_truth = np.asarray(ground_truth_crops)

        # Convert to float32
        N_image, radar_points = [
            T.astype(np.float32)
            for T in [N_image, radar_points]
        ]

        bounding_boxes_list = [T.astype(np.float32) for T in bounding_boxes_list]
        # print("this bounding_boxes_list is ", bounding_boxes_list)
        bounding_boxes_list = np.stack(bounding_boxes_list, axis=0)
        # assert len(bounding_boxes_list) == radar_points.shape[0]
        # #读lidar数据
    
        # print(lidar_path)
        # lidar_path = lidar_path.replace(".png", ".bin")
        # lidar_point = self.load_lidar(lidar_path)
        
        if "plane_params" in self.data:
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
            )
        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()

        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        xy_w_gt = self.data["t_c2w"][idx][:2].numpy()
        uv_gt = canvas.to_uv(xy_w_gt)
        uv_init = canvas.to_uv(bbox_tile.center)
        raster = canvas.raster  # C, H, W

        image = N_image
        camera_image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed 
        )  

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = yaw + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][-2:].numpy()
            xy_gps = self.tile_managers[scene].projection.project(gps)
            data["uv_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
            data["accuracy_gps"] = torch.tensor(
                min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        return {
            **data,
            # "image": image,
            "image": camera_image,
            "N_image": N_image,
            # "radar_image":radar_image,
            # "lidar":lidar_point,
            "radar_points": radar_points,
            "bounding_boxes_list": bounding_boxes_list,
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "uv": torch.from_numpy(uv_gt).float(),  # TODO: maybe rename to uv?
            "uv_init": torch.from_numpy(uv_init).float(),  # TODO: maybe rename to uv?
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }
    # def get_view(self, idx, scene, seq, name, seed, bbox_tile):
    #     data = {
    #         "index": idx,
    #         "name": name,
    #         "scene": scene,
    #         "sequence": seq,
    #     }
    #     cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
    #     cam = Camera.from_dict(cam_dict).float()
    #     #记录对应的map
    #     output_txt_path = "mapandradar.txt"
    #     xmin, ymin, xmax, ymax = bbox_tile.min_[0], bbox_tile.min_[1], bbox_tile.max_[0], bbox_tile.max_[1]
    #     bbox_tile_str = f"{xmin},{ymin},{xmax},{ymax}"
    #     image_path = self.image_dirs[scene] / (name + self.image_ext)
    #     # with open(output_txt_path, 'a') as txt_file:
    #     #     txt_file.write(f"{bbox_tile_str} {image_path}\n")

    #     roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()

    #     image = read_image(self.image_dirs[scene] / (name + self.image_ext))

    #     image_path = str(self.image_dirs[scene] / (name + self.image_ext))
    #     #读相机的图像
    #     camera_image_path = image_path.replace("/radar/cart", "/camera_new")
    #     camera_image = read_image(Path(camera_image_path))
    #     #读雷达的图像
    #     radar_image_path = image_path
    #     radar_image = read_image(Path(radar_image_path))


    #     #单独的读图###
    #     # radar_image_path = image_path.replace("/cart", "/cart_orignal")
    #     # rotated_image_path = image_path.replace("/cart", "/cart_rotated")
    #     # radar_image = Image.open(radar_image_path)
    #     # tensor_image  = self.read_Image(radar_image_path)
    #     # tensor_rotated_image = self.read_Image(rotated_image_path)
    #     # radar_pil = ToPILImage()(tensor_image.cpu())
    #     # radar_pil.save("radar_image11.png")s
    #     # radar_pil = ToPILImage()(tensor_rotated_image.cpu())
    #     # radar_pil.save("rotated_image11.png")
    #     # print(image_path)
    #     # rotated_image = read_image(Path(rotated_image_path))
    #     # rotated_image = cv2.resize(rotated_image, (256, 256))


    #     # 矫正相机到水平 radar中去掉
    #     # if "plane_params" in self.data:
    #     #     # transform the plane parameters from world to camera frames
    #     #     plane_w = self.data["plane_params"][idx]
    #     #     data["ground_plane"] = torch.cat(
    #     #         [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
    #     #     )
    #     if self.cfg.force_camera_height is not None:
    #         data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
    #     elif "camera_height" in self.data:
    #         data["camera_height"] = self.data["height"][idx].clone()

    #     # raster extraction
    #     fig, axes = plt.subplots()
    #     canvas = self.tile_managers[scene].query(bbox_tile)
    #     map_v = Colormap.apply_lines(canvas.raster)
    #     map_viz=plot_images([map_v], titles=["OpenStreetMap raster"])#[256,256,3]       
    #     map_viz= torch.tensor(np.array(map_viz[0])).permute(2, 0, 1)
    #     # 将map_viz转换为PIL Image对象  
    #     # Convert map_tensor data type to uint8


    #     # 保存图片到本地  
    #     # print("map_viz Shape:", map_viz.shape)
    #     plt.close()


    #     xy_w_gt = self.data["t_c2w"][idx][:2].numpy()#相机到世界坐标系的转换矩阵
    #     uv_gt = canvas.to_uv(xy_w_gt)#将世界坐标系中的坐标 xy_w_gt 转换为画布坐标系中的 UV 坐标
    #     uv_init = canvas.to_uv(bbox_tile.center)
    #     raster = canvas.raster  # C, H, W
    #     heading = np.deg2rad(90 - yaw)  # fixme
    #     # print("heading(degree):",heading)
    #     # if self.stage == "train":
    #     #     if self.cfg.augmentation.rot90:
    #     #         raster, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed)
    #     #     if self.cfg.augmentation.flip:
    #     #         image, raster, uv_gt, heading = random_flip(
    #     #             image, raster, uv_gt, heading, seed
    #     #         )
    #     # yaw = 90 - np.rad2deg(heading)  # fixme
        
    #     # print("yaw(degree):",heading)
    #     # 获取旋转矩阵
    #     # rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), yaw, 1)
    #     # # 旋转雷达图像
    #     # rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    #     # # 将NumPy数组转换为PIL Image对象
    #     # rotated_image_pil = Image.fromarray(rotated_image)

    #     # # 保存旋转后的雷达图像
    #     # rotated_image_path = "radar_rotated.png"  # 替换成实际保存路径
    #     # rotated_image_pil.save(rotated_image_path)
    #     # 将雷达图像叠加到地图上

    #     # 保存map图像
    #     # image_map = torch.tensor(map_viz).permute(1, 2, 0)
    #     image_map = map_viz.clone().detach().permute(1, 2, 0)
    #     image_map = image_map.numpy()   
    #     image_map = np.array(image_map)     
    #     image_map = image_map.astype(np.uint8)
    #     image_map_pil = Image.fromarray(image_map)
    #     image_map_pil.save("image_map.png")



    #     #这里图像换成点云
    #     # image_map_pil.save("image_map_with_radar.png")

    #     image = camera_image
    #     camera_image, valid, cam, roll, pitch = self.process_image(
    #         image, cam, roll, pitch, seed
    #     )#根据roll pitch和cam做矫正【3，256，256】
    #     # max_value = np.max(rotated_image)
    #     # rotated_image = (
    #     #     torch.from_numpy(np.ascontiguousarray(rotated_image))
    #     #     .permute(2, 0, 1)
    #     #     .float()
    #     #     .div_(1)
    #     # )
    #     # is_all_zeros = torch.all(rotated_image.eq(0))
    #     # print("Is rotated_image all zeros?", is_all_zeros)
    #     # Create the mask for prior location
    #     image = radar_image
    #     radar_image, valid_radar, cam, roll, pitch = self.process_image(
    #         image, cam, roll, pitch, seed
    #     )
    #     if self.cfg.add_map_mask:
    #         data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

    #     if self.cfg.max_init_error_rotation is not None:
    #         if "shifts" in self.data:
    #             error = self.data["shifts"][idx][-1]
    #         else:
    #             error = np.random.RandomState(seed + 1).uniform(-1, 1)
    #             error = torch.tensor(error, dtype=torch.float)
    #         yaw_init = yaw + error * self.cfg.max_init_error_rotation
    #         range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
    #         data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

    #     if self.cfg.return_gps:
    #         gps = self.data["gps_position"][idx][:2].numpy()
    #         xy_gps = self.tile_managers[scene].projection.project(gps)
    #         data["uv_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
    #         data["accuracy_gps"] = torch.tensor(
    #             min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
    #         )

    #     if "chunk_index" in self.data:
    #         data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])
    #     # print(torch.from_numpy(np.ascontiguousarray(raster)).long().shape)
    #     return {
    #         **data,
    #         # "image": tensor_image,
    #         "image": camera_image,
    #         "radar_image":radar_image,
    #         # "rotated_image":tensor_rotated_image,
    #         "valid": valid,
    #         "camera": cam,
    #         "canvas": canvas,
    #         "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
    #         "map_viz":map_viz,
    #         "uv": torch.from_numpy(uv_gt).float(),  # TODO: maybe rename to uv?
    #         "uv_init": torch.from_numpy(uv_init).float(),  # TODO: maybe rename to uv?
    #         "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
    #         "pixels_per_meter": torch.tensor(canvas.ppm).float(),
    #     }
    def calculate_single_channel_mean(self,image, channel_index):
        # 打开图像
        # 获取图像统计信息
        stat = ImageStat.Stat(image)
        # 获取指定通道的均值
        mean_value = stat.mean[channel_index]
        return mean_value
    def get_bin_table(self,threshold=10):
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(i)
        return table
    def read_Image(self,image_path):
        radar_image = Image.open(image_path)
        radar_image= radar_image.resize((256, 256))
        imgry = radar_image.convert('L')
        channel_index = 0  # 替换为你需要的通道索引
        # 计算指定通道的均值
        mean_value = self.calculate_single_channel_mean(imgry, channel_index)
        table = self.get_bin_table(threshold=mean_value+10)
        binary = imgry.point(table)
        # binary = imgry.point(table)
        binary.save('binary.png')
        numpy_array = np.array(binary)
        tensor_image = torch.from_numpy(numpy_array)
        tensor_image = tensor_image.float()
        tensor_image[tensor_image > 0] = 1
        return tensor_image


    def process_image(self, image, cam, roll, pitch, seed):
        max_value = np.max(image)
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            # .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        # image, valid = rectify_image(
        #     image, cam, roll, pitch if self.cfg.rectify_pitch else None
        # )
        valid = torch.ones_like(image[0], dtype=torch.bool)
        roll = 0.0
        if self.cfg.rectify_pitch:
            pitch = 0.0

        # if self.cfg.target_focal_length is not None:
        #     # resize to a canonical focal length
        #     factor = self.cfg.target_focal_length / cam.f.numpy()
        #     size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
        #     # size=int(size)
        #     # image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
        #     size_out = self.cfg.resize_image
        #     if size_out is None:
        #         # round the edges up such that they are multiple of a factor
        #         stride = self.cfg.pad_to_multiple
        #         size_out = (np.ceil((size / stride)) * stride).astype(int)
        #     # crop or pad such that both edges are of the given size
        #     image, valid, cam = pad_image(
        #         image, size_out, cam, valid, crop_and_center=True
        #     )
        # if self.cfg.resize_image is not None:
        #     image, _, cam, valid = resize_image(
        #         image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
        #     )
        #     if self.cfg.pad_to_square:
        #         # pad such that both edges are of the given size
        #         image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)

        # if self.cfg.reduce_fov is not None:
        #     h, w = image.shape[-2:]
        #     f = float(cam.f[0])
        #     fov = np.arctan(w / f / 2)
        #     w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
        #     image, valid, cam = pad_image(
        #         image, (w_new, h), cam, valid, crop_and_center=True
        #     )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        return image, valid, cam, roll, pitch

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask
