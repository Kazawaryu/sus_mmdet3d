# model settings (based on nuScenes model settings)
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.



voxel_size = [0.3125, 0.3125, 8]
model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[-80, -80, -5, 80, 80, 3],
            voxel_size=voxel_size,
            max_voxels=(50000, 50000))),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=7,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-80, -80, -5, 80, 80, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', 
        in_channels=64, 
        output_shape=[512, 512]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,

        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                    [-80, -80, -1.03112055, 80, 80, -1.03112055],
                    [-80, -80, -0.86299253, 80, 80, -0.86299253],
                    [-80, -80, -0.99646653, 80, 80, -0.99646653],
                    [-80, -80, -0.29750652, 80, 80, -0.29750652],
                    [-80, -80, -1.11042015, 80, 80, -1.11042015],

                    [-80, -80, -1.16544953, 80, 80, -1.16544953],
                    [-80, -80, -0.81176071, 80, 80, -0.81176071],
                    [-80, -80, -0.2890795,  80, 80, -0.2890795],
                    [-80, -80, -0.95039066, 80, 80, -0.95039066],
                    [-80, -80, -0.98711163, 80, 80, -0.98711163],
            ],
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
                    [2.85369005, 1.23022373, 1.64355945],
            ],
            rotations=[0, 1.57],
            reshape_out=False),

        # anchor_generator=dict(
        #     type='AlignedAnchor3DRangeGenerator',
        #     ranges=[[-80, -80, -1.8, 80, 80, -1.8]],
        #     scales=[1, 2, 4],
        #     sizes=[
        #         [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
        #         [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
        #         [1., 1., 1.],
        #         [0.4, 0.4, 1],
        #     ],
        #     custom_values=[],
        #     rotations=[0, 1.57],
        #     reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
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
        pts=dict(
            assigner=dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=4096,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))


