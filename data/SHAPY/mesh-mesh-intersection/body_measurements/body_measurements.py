# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE ĐO LƯỜNG CƠ THỂ 3D THEO CHUẨN ISO 8559 (SHAPY BODY MEASUREMENTS)
-------------------------------------------------------------------------------
[INDEX CÁC PHƯƠNG THỨC TRONG FILE]:
1. __init__(cfg)                        : Đọc file định nghĩa mốc giải phẫu (.yaml).
2. _intersect_horizontal_plane_pytorch(): Cắt lát mặt phẳng ngang Y = h + Lọc bỏ cánh tay T-pose.
3. compute_peripheries()                : Tính chu vi Ngực, Eo, Hông bằng ConvexHull 2D.
4. compute_height()                     : Đo khoảng cách trục Y từ Đỉnh đầu đến Gót chân trái.
5. compute_mass()                       : Tính thể tích lưới 3D x Khối lượng riêng (985 kg/m3).
6. forward()                            : Trả về dict chứa 'mass', 'height', 'chest', 'waist', 'hips'.
===============================================================================
"""

from typing import NewType, Dict, Tuple
import os.path as osp
import yaml
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import ConvexHull

Tensor = NewType('Tensor', torch.Tensor)


class BodyMeasurements(nn.Module):
    # Khối lượng riêng trung bình của cơ thể người: 985 kg/m^3
    DENSITY = 985

    def __init__(self, cfg, **kwargs):
        super(BodyMeasurements, self).__init__()

        meas_definition_path = osp.expanduser(osp.expandvars(cfg.get('meas_definition_path', '')))
        meas_vertices_path = osp.expanduser(osp.expandvars(cfg.get('meas_vertices_path', '')))

        with open(meas_definition_path, 'r') as f:
            measurements_definitions = yaml.safe_load(f)

        with open(meas_vertices_path, 'r') as f:
            meas_vertices = yaml.safe_load(f)

        head_top = meas_vertices['HeadTop']
        left_heel = meas_vertices['HeelLeft']

        self.left_heel_face_idx = left_heel['face_idx']
        self.register_buffer('left_heel_bc', torch.tensor(left_heel['bc'], dtype=torch.float32))

        self.head_top_face_idx = head_top['face_idx']
        self.register_buffer('head_top_bc', torch.tensor(head_top['bc'], dtype=torch.float32))

        chest_data = meas_vertices[measurements_definitions['CW_p'][0]]
        self.chest_face_index = chest_data['face_idx']
        self.register_buffer('chest_bcs', torch.tensor(chest_data['bc'], dtype=torch.float32))

        belly_data = meas_vertices[measurements_definitions['BW_p'][0]]
        self.belly_face_index = belly_data['face_idx']
        self.register_buffer('belly_bcs', torch.tensor(belly_data['bc'], dtype=torch.float32))

        hips_data = meas_vertices[measurements_definitions['IW_p'][0]]
        self.hips_face_index = hips_data['face_idx']
        self.register_buffer('hips_bcs', torch.tensor(hips_data['bc'], dtype=torch.float32))

    @staticmethod
    def _intersect_horizontal_plane_pytorch(triangles_single: Tensor, plane_y: float, filter_arms: bool = True) -> np.ndarray:
        """
        Tìm giao điểm giữa lưới tam giác 3D (Fx3x3) và mặt phẳng ngang Y = plane_y.
        Nếu filter_arms=True: Tự động loại bỏ các điểm thuộc 2 cánh tay/bàn tay đang dang ngang ở T-pose
        bằng cách phát hiện khoảng hở không khí (gap) giữa lồng ngực và cánh tay dọc theo trục X.
        """
        y = triangles_single[:, :, 1] - plane_y
        edges = [(0, 1), (1, 2), (2, 0)]
        intersect_pts = []

        for i0, i1 in edges:
            y0, y1 = y[:, i0], y[:, i1]
            mask = ((y0 <= 0) & (y1 >= 0)) | ((y0 >= 0) & (y1 <= 0))
            valid = mask & (torch.abs(y1 - y0) > 1e-8)
            if valid.any():
                t = (-y0[valid] / (y1[valid] - y0[valid])).unsqueeze(-1)
                v0 = triangles_single[valid, i0, :]
                v1 = triangles_single[valid, i1, :]
                pts = v0 + t * (v1 - v0)
                intersect_pts.append(pts)

        if len(intersect_pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pts_np = torch.cat(intersect_pts, dim=0).detach().cpu().numpy()

        if filter_arms and len(pts_np) > 10:
            # Lồng ngực luôn đối xứng quanh tâm X = 0.
            # Cánh tay giơ ngang ở T-pose chỉ xuất hiện khi có điểm vươn xa (|X| >= 0.28m).
            x_abs = np.abs(pts_np[:, 0])
            if np.max(x_abs) >= 0.28:
                sorted_x = np.sort(x_abs)
                diffs = np.diff(sorted_x)
                # Khoảng trống không khí thực tế giữa nách/thân người và bắp tay T-pose luôn >= 3.5cm (0.035m).
                # Chỉ quét tìm khoảng trống bắt đầu từ ngoài bán kính thân người tối thiểu (> 0.16m).
                gap_indices = np.where((sorted_x[:-1] > 0.16) & (diffs > 0.035))[0]
                if len(gap_indices) > 0:
                    cutoff_x = sorted_x[gap_indices[0]] + 0.01
                    pts_np = pts_np[x_abs <= cutoff_x]
                elif np.max(x_abs) > 0.40:
                    # Nếu cánh tay giơ ngang xa (> 0.40m) mà mesh nối liền nách, cắt ở biên sườn an toàn
                    pts_np = pts_np[x_abs <= 0.32]

        return pts_np

    def compute_peripheries(
        self,
        triangles: Tensor,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
    ) -> Dict[str, Tensor]:
        batch_size = triangles.shape[0]
        device = triangles.device

        meas_data = {}
        if compute_chest:
            # Chỉ lọc cánh tay ở vòng ngực (chest). Ở T-pose, eo và hông nằm sâu bên dưới, không bao giờ chạm tay.
            meas_data['chest'] = (self.chest_face_index, self.chest_bcs, True)
        if compute_waist:
            meas_data['waist'] = (self.belly_face_index, self.belly_bcs, False)
        if compute_hips:
            meas_data['hips'] = (self.hips_face_index, self.hips_bcs, False)

        output = {}
        for name, (face_index, bcs, should_filter_arms) in meas_data.items():
            vertex = (triangles[:, face_index] * bcs.reshape(1, 3, 1)).sum(axis=1)

            output[name] = {
                'points': [],
                'valid_points': [],
                'value': [],
                'plane_height': vertex[:, 1],
            }

            for ii in range(batch_size):
                plane_y = float(vertex[ii, 1].item())
                pts_3d = self._intersect_horizontal_plane_pytorch(
                    triangles[ii], plane_y, filter_arms=should_filter_arms
                )

                points_in_plane = pts_3d[:, [0, 2]]
                hull = ConvexHull(points_in_plane)
                point_indices = hull.simplices.reshape(-1)

                pts_tensor = torch.tensor(pts_3d, device=device, dtype=triangles.dtype)
                hull_points = pts_tensor[point_indices].reshape(-1, 2, 3)

                meas_value = (hull_points[:, 1] - hull_points[:, 0]).pow(2).sum(dim=-1).sqrt().sum()

                output[name]['valid_points'].append(pts_3d)
                output[name]['points'].append(hull_points)
                output[name]['value'].append(meas_value)

            output[name]['tensor'] = torch.stack(output[name]['value'])
        return output

    def compute_height(self, shaped_triangles: Tensor) -> Tuple[Tensor, Tensor]:
        head_top_tri = shaped_triangles[:, self.head_top_face_idx]
        head_top = (head_top_tri * self.head_top_bc.reshape(1, 3, 1)).sum(dim=1)

        left_heel_tri = shaped_triangles[:, self.left_heel_face_idx]
        left_heel = (left_heel_tri * self.left_heel_bc.reshape(1, 3, 1)).sum(dim=1)

        return (
            torch.abs(head_top[:, 1] - left_heel[:, 1]),
            torch.stack([head_top, left_heel], axis=0)
        )

    def compute_mass(self, tris: Tensor) -> Tensor:
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def forward(
        self,
        triangles: Tensor,
        compute_mass: bool = True,
        compute_height: bool = True,
        compute_chest: bool = True,
        compute_waist: bool = True,
        compute_hips: bool = True,
        **kwargs
    ):
        measurements = {}
        if compute_mass:
            measurements['mass'] = {'tensor': self.compute_mass(triangles)}

        if compute_height:
            mesh_height, points = self.compute_height(triangles)
            measurements['height'] = {'tensor': mesh_height, 'points': points}

        output = self.compute_peripheries(
            triangles,
            compute_chest=compute_chest,
            compute_waist=compute_waist,
            compute_hips=compute_hips,
        )
        measurements.update(output)
        return {'measurements': measurements}
