_base_ = [
    '../_base_/models/point_rcnn.py',
    "../_base_/datasets/suscape-3d.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cyclic-20e.py"
]

point_cloud_range = [-80, -80, -5, 80, 80, 3]

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

input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

model = dict(
    rpn_head=dict(
        num_classes=len(_base_["class_names"]),
        bbox_coder=dict(
            use_mean_size=True,
            mean_size=[
                [4.42419589, 1.974167, 1.63783862],
                [0.65047053, 0.66835484, 1.6406836],
                [1.71321554, 0.75332566, 1.59110313],
                [8.309413, 2.62914506, 3.06056424],
                [1.64555086, 0.66646036, 1.22126398],
                [1.57330043, 0.57421994, 1.14353524],
                [4.57143242, 1.98126929, 2.02480982],
                [10.34199791, 2.96619499, 3.22222376],
                [1.66885451, 0.68840352, 1.68261234],
                [2.85369005, 1.23022373, 1.64355945],
            ],
        ),
    )
)

dataset_type = "SuscapeDataset"
data_root = "data/suscape/"

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="PointSample", num_points=16384, sample_range=None),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="PointSample", num_points=16384, sample_range=None),
    dict(type="Pack3DDetInputs", keys=["points"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="suscape_infos_train.pkl",
            data_prefix=dict(pts=""),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d="LiDAR",
        ),
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=""),
        ann_file="suscape_infos_val.pkl",
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=""),
        ann_file="suscape_infos_val.pkl",
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
    ),
)

val_evaluator = dict(
    type="SuscapeMetric", data_root="./", ann_file=data_root + "suscape_infos_val.pkl"
)
test_evaluator = val_evaluator


lr = 0.0001
optim_wrapper = dict(optimizer=dict(lr=lr, betas=(0.95, 0.85)))
num_epochs = 40
train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=20)

auto_scale_lr = dict(enable=False, base_batch_size=32)
param_scheduler = [
    # learning rate scheduler
    # During the first 35 epochs, learning rate increases from 0 to lr * 10
    # during the next 45 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=num_epochs * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=num_epochs * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=num_epochs * 0.6,
        eta_min=lr * 1e-4,
        begin=num_epochs * 0.4,
        end=num_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first 35 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 45 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=num_epochs * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=num_epochs * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=num_epochs * 0.6,
        eta_min=1,
        begin=num_epochs * 0.4,
        end=num_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]