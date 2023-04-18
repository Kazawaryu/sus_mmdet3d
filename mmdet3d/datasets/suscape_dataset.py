# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class SuscapeDataset(Det3DDataset):
    r"""SUSCape Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    
    METAINFO = {
        'classes': ['Car', 'Pedestrian', 'ScooterRider']# ('Car', 'Truck', 'Bus', 'Bicycle', 'Scooter', 'ScooterRider')
    }
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs):
        assert box_type_3d.lower() in ['lidar']
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of 3D ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            # empty instance
            anns_results = dict()
            anns_results['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            anns_results['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            return anns_results
        
        ann_info = self._remove_dontcare(ann_info)
        
        gt_bboxes_3d = ann_info['gt_bboxes_3d']
        gt_labels_3d = ann_info['gt_labels_3d']

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        return anns_results

    # def get_data_info(self, index):
    #     """Get data info according to the given index.

    #     Args:
    #         index (int): Index of the sample data to get.

    #     Returns:
    #         dict: Data information that will be passed to the data
    #             preprocessing pipelines. It includes the following keys:

    #             - sample_idx (str): Sample index.
    #             - pts_filename (str): Filename of point clouds.
    #             - img_prefix (str): Prefix of image files.
    #             - img_info (dict): Image info.
    #             - lidar2img (list[np.ndarray], optional): Transformations
    #                 from lidar to different cameras.
    #             - ann_info (dict): Annotation info.
    #     """
    #     info = self.data_infos[index]


    #     pts_filename = info["lidar_path"]
    #     input_dict = dict(            
    #         pts_filename=pts_filename,
    #         )

    #     if not self.test_mode:
    #         annos = self.get_ann_info(index)
    #         input_dict['ann_info'] = annos

    #     return input_dict

    # def get_ann_info(self, index):
    #     """Get annotation info according to the given index.

    #     Args:
    #         index (int): Index of the annotation data to get.

    #     Returns:
    #         dict: annotation information consists of the following keys:

    #             - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
    #                 3D ground truth bboxes.
    #             - gt_labels_3d (np.ndarray): Labels of ground truths.
    #             - gt_bboxes (np.ndarray): 2D ground truth bboxes.
    #             - gt_labels (np.ndarray): Labels of ground truths.
    #             - gt_names (list[str]): Class names of ground truths.
    #             - difficulty (int): Difficulty defined by KITTI.
    #                 0, 1, 2 represent xxxxx respectively.
    #     """

    #     # Use index to get the annos, thus the evalhook could also use this api
    #     info = self.data_infos[index]
        
    #     # print("get info index", index, info["lidar_path"])
    #     # should we remove some object types?
        
    #     gt_names = info['gt_names']
    #     gt_bboxes_3d = np.array(info['gt_boxes']).astype(np.float32)

    #     if gt_bboxes_3d.shape[0] == 0:
    #         print(index, gt_bboxes_3d.shape)
    #     # gt_bboxes = annos['bbox']
    #     gt_bboxes_3d = LiDARInstance3DBoxes(
    #                 gt_bboxes_3d,
    #                 box_dim=gt_bboxes_3d.shape[-1],
    #                 origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

    #     gt_labels = []
    #     for cat in gt_names:
    #         if cat in self.CLASSES:
    #             gt_labels.append(self.CLASSES.index(cat))
    #         else:
    #             gt_labels.append(-1)
    #     gt_labels = np.array(gt_labels).astype(np.int64)
    #     gt_labels_3d = copy.deepcopy(gt_labels)

    #     anns_results = dict(
    #         gt_bboxes_3d=gt_bboxes_3d,
    #         gt_labels_3d=gt_labels_3d,
    #         # bboxes=gt_bboxes,
    #         # labels=gt_labels,
    #         gt_names=gt_names,
    #         # plane=plane_lidar,
    #         # difficulty=difficulty
    #         )
    #     return anns_results
