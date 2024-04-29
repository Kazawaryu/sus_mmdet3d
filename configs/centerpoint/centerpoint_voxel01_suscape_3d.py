# """configs/centerpoint/centerpoint_pillar03_kitti_3d.py"""
_base_ = [
    "../_base_/datasets/suscape-3d.py",
    "../_base_/models/centerpoint_voxel01_second_secfpn_kitti.py",
    "../_base_/schedules/cyclic-20e.py",
    "../_base_/default_runtime.py",
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-80, -80, -5, 80, 80, 3]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]

# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
class_names = [
    "Car",
    "Pedestrian",
    "ScooterRider",
    "Truck",
    "Scooter",
    "Bicycle",
    "Van",
    "Bus",
    "BicycleRider",  #'BicycleGroup',
    "Trimotorcycle",  #'RoadWorker',
]
# data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
model = dict(
    data_preprocessor=dict(voxel_layer=dict(point_cloud_range=point_cloud_range)),
    # pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])),
)

# dataset_type = 'NuScenesDataset'
# data_root = 'data/nuscenes/'
dataset_type = "SuscapeDataset"
data_root = "data/suscape/"
backend_args = None

# db_sampler = dict(
#     data_root=data_root,
#     # info_path=data_root + 'nuscenes_dbinfos_train.pkl',
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     # prepare=dict(
#     #     filter_by_difficulty=[-1],
#     #     filter_by_min_points=dict(
#     #         car=5,
#     #         truck=5,
#     #         bus=5,
#     #         trailer=5,
#     #         construction_vehicle=5,
#     #         traffic_cone=5,
#     #         barrier=5,
#     #         motorcycle=5,
#     #         bicycle=5,
#     #         pedestrian=5)),
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(
#             car=5,
#             Cyclist = 5,
#             pedestrian=5)),
#     classes=class_names,
#     sample_groups=dict(
#             car=2,
#             Cyclist = 4,
#             pedestrian=4),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         # load_dim=5,
#         load_dim=4,
#         # use_dim=[0, 1, 2, 3, 4],
#         backend_args=backend_args),
#     backend_args=backend_args)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        # load_dim=5,
        # use_dim=5,
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(
        type="RandomFlip3D",
        # sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        # flip_ratio_bev_vertical=0.5
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        # load_dim=5,
        # use_dim=5,
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        ],
    ),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

# train_dataloader = dict(
#     _delete_=True,
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             # ann_file='nuscenes_infos_train.pkl',
#             ann_file='kitti_infos_train.pkl',
#             pipeline=train_pipeline,
#             metainfo=dict(classes=class_names),
#             test_mode=False,
#             data_prefix=data_prefix,
#             use_valid_flag=True,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR',
#             backend_args=backend_args)))


train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, metainfo=dict(classes=class_names))
    ),
    # dataset=dict(pipeline=train_pipeline, metainfo=dict(classes=class_names)),
)


test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)),
)


lr = 0.0001
epoch_num = 40
optim_wrapper = dict(optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=20)
val_cfg = dict()

# train_cfg = dict(val_interval=20)
