_base_ = [
    '../_base_/models/pointpillars_hv_fpn_suscape.py',
    '../_base_/datasets/suscape-3d.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]

backend_args = None

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
)


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=2)

# model settings
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=[-80, -80, -5, 80, 80, 3])),
    pts_voxel_encoder=dict(
        feat_channels=[32, 64],
        point_cloud_range=[-80, -80, -5, 80, 80, 3]),
    pts_middle_encoder=dict(output_shape=[800, 800]),
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        _delete_=True,
        type='ShapeAwareHead',
        num_classes=9,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGeneratorPerCls',
            ranges=[[-80, -80, -1.03112055, 80, 80, -1.03112055],
                    [-80, -80, -0.86299253, 80, 80, -0.86299253],
                    [-80, -80, -0.99646653, 80, 80, -0.99646653],
                    [-80, -80, -0.29750652, 80, 80, -0.29750652],
                    [-80, -80, -1.11042015, 80, 80, -1.11042015],

                    [-80, -80, -1.16544953, 80, 80, -1.16544953],
                    [-80, -80, -0.81176071, 80, 80, -0.81176071],
                    [-80, -80, -0.2890795, 80, 80, -0.2890795],
                    [-80, -80, -0.95039066, 80, 80, -0.95039066],
                    [-80, -80, -0.98711163, 80, 80, -0.98711163]],
            sizes=[
                    [4.42419589, 1.974167  , 1.63783862],
                    [0.65047053, 0.66835484, 1.6406836 ],
                    [1.71321554, 0.75332566, 1.59110313],
                    [8.309413  , 2.62914506, 3.06056424],
                    [1.64555086, 0.66646036, 1.22126398],

                    [1.57330043, 0.57421994, 1.14353524],
                    [4.57143242, 1.98126929, 2.02480982],
                    [10.34199791,2.96619499, 3.22222376],
                    [1.66885451, 0.68840352, 1.68261234],
                    [2.85369005, 1.23022373, 1.64355945]],
            custom_values=[],
            rotations=[0, 1.57],
            reshape_out=False),
        tasks=[
            dict(
                num_class=2,
                class_names=['Bicycle', 'Scooter'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=3,
                class_names=['BicycleRider', 'ScooterRider', 'Trimotorcycle'],
                shared_conv_channels=(64, 64, 64),
                shared_conv_strides=(1, 1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=1,
                class_names=['Pedestrian'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(1, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=2,
                class_names=['Car', 'Van'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(2, 1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01)),
            dict(
                num_class=2,
                class_names=['Bus', 'Truck'],
                shared_conv_channels=(64, 64),
                shared_conv_strides=(2,  1),
                norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01))
        ],
        assign_per_class=True,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            assigner=[
                dict(  # Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # scooterrider
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # scooter
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # bicycle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # van
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # byciclerider
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # trimotorcycle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                
                
            ],
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
