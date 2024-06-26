# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize
import torchvision
from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
import os
from .bev_projection import CartesianProjection, PolarProjectionDepth
from PIL import Image
from .voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
import matplotlib.pyplot as plt
import numpy as np
from .radarnet_model import RadarNetModel
import torch.nn.functional as F
class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
        "radarnet_model": "???",
        "input_patch_size_image":(512, 64),
        "output_depth_dirpath": "/home/classlab2/root/OrienterNet/depth_output",
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)
        self.radar_depth = RadarNetModel(conf.radarnet_model)


    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores
    def save_depth(self, z, path, multiplier=256.0):
        '''
        Saves a depth map to a 16-bit PNG file

        Arg(s):
            z : numpy[float32]
                depth map
            path : str
                path to store depth map
            multiplier : float
                multiplier for encoding float as 16/32 bit unsigned integer
        '''

        z = np.uint32(z * multiplier)
        z = Image.fromarray(z, mode='I')
        z.save(path)

    def _forward(self, data):
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # Extract image features.
        level = 0
        #引入融合的depth估计
        depth = pred["depth_prior"] = self.radar_depth(data)#[930,1,256,256]
        #对雷达点估计的logits恢复为深度图
        patch_size = self.conf.input_patch_size_image
        pad_size = patch_size[1] // 2
        image = data["N_image"]#[1,3,256,320]
        radar_points = data["radar_points"]#[1,930,3]
        image = torchvision.transforms.functional.pad(
        image,
        (pad_size, 0, pad_size, 0),
        padding_mode='edge')#[1,3,256,576]
        start_y = image.shape[-2] - patch_size[0]
        output_tiles = []
        if radar_points.dim() == 3:
            # 将雷达点从 1 x N x 3 转换为 N x 3
            radar_points = torch.squeeze(radar_points, dim=0)#[930,3]
        x_shifts = radar_points[:, 0].clone()

        height = image.shape[-2]
        crop_height = height - patch_size[0]
        output_crops = depth

        for output_crop, x in zip(output_crops, x_shifts):
            output = torch.zeros([1, image.shape[-2], image.shape[-1]])

            # 对任何小于0.5的响应进行阈值处理，将其设置为0
            output_crop = torch.where(output_crop < 0.5, torch.zeros_like(output_crop), output_crop)
            # 将裁剪区域添加到输出
            # output[:, crop_height:, int(x)-pad_size:int(x)+pad_size] = output_crop
            # print(crop_height)
            # print(int(x))
            # print(int(x) - pad_size)
            # print(output_crop)
            output[:, crop_height:, int(x)-pad_size:int(x)+pad_size] = output_crop
            output_tiles.append(output)

        # 将所有裁剪区域拼接起来
        output_tiles = torch.cat(output_tiles, dim=0)
        output_tiles = output_tiles[:, :, pad_size:-pad_size]

        # 在所有裁剪区域中找到最大响应
        output_response, output = torch.max(output_tiles, dim=0, keepdim=True)

        # 根据所选择点的z值填充地图
        for point_idx in range(radar_points.shape[0]):
            output = torch.where(
                output == point_idx,
                torch.full_like(output, fill_value=radar_points[point_idx, 2]),
                output)


        # 如果没有预测到结果，则保持为0
        output_depth = torch.where(
            torch.max(output_tiles, dim=0, keepdim=True)[0] == 0,
            torch.zeros_like(output),
            output)
        # print(output_depth.shape)
        # output_depth = np.squeeze(output_depth.cpu().numpy())
        depth_tensor = output_depth.unsqueeze(0)#[1,1,512,676]
    
        depth_tensor = depth_tensor.to(torch.float32).to(data["image"].device)
        data["N_image"] = data["N_image"].to(data["image"].device)
        # 调整张量大小
        # resized_tensor = F.interpolate(depth_tensor, size=(256, 256), mode='bilinear', align_corners=False)

        # # 打印结果形状
        # print(resized_tensor.shape)  # 输出: torch.Size([1, 1, 256, 256])
        image_d = torch.cat((data["N_image"], depth_tensor), dim=1).to(data["image"].device)
        data["image_d"] = image_d

        f_image = self.image_encoder(data)["feature_maps"][level]#[1,128,128,128]
        

        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # if np.all(output_depth == 0):
        #     print("output_depth 全为零")
        # else:
        #     print("output_depth 有有效值")
        # #保存深度图
        filename = data['name']
        filename = os.path.basename(filename[0])
        output_depth_path = os.path.join(self.conf.output_depth_dirpath, filename)
        depth_img = np.squeeze(output_depth.cpu().numpy())
        self.save_depth(depth_img, output_depth_path)

        # print(depth.shape)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))#【1，128，128，33】
        # print(scales.shape)
        # self.visualize_depth_distribution_heatmap(scales)
        f_polar = self.projection_polar(f_image, scales, camera)#[torch.Size([1, 128, 64, 128])]
        # print(f_polar.shape)
        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]

        scores = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        # if "map_mask" in data:
        #     scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        # if "yaw_prior" in data:
        #     mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }