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
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="PointSample", num_points=16384, sample_range=None),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

train_dataloader = dict(
    batch_size=10,
    num_workers=4,
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, metainfo=dict(classes=class_names))
    ),
    # dataset=dict(pipeline=train_pipeline, metainfo=dict(classes=class_names)),
)


test_dataloader = dict(
    batch_size=10,
    num_workers=4,
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)),
)
val_dataloader = dict(
    batch_size=10,
    num_workers=4,
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