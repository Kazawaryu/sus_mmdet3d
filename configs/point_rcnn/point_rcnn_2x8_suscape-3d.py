_base_ = [
    '../_base_/datasets/suscape-3d.py', '../_base_/models/point_rcnn.py',
    '../_base_/default_runtime.py', '../_base_/schedules/cyclic-40e.py'
]

# dataset settings
dataset_type = 'SuscapeDataset'
data_root = 'data/suscape/'
class_names = ['Car', 'Pedestrian', 'ScooterRider'
               , 'Truck', 'Scooter',
                'Bicycle', 'Van', 'Bus', 'BicycleRider', #'BicycleGroup', 
                'Trimotorcycle', #'RoadWorker', 
                ]

point_cloud_range = [-80, -80, -5, 80, 80, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=class_names,
#     sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
#     points_loader=dict(
#         type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),    
    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0.5],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointSample', num_points=16384, sample_range=None),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointShuffle'),
    # dict(
    #     type='PointsSave',
    #     path='./temp'
    # ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='PointSample', num_points=16384, sample_range=None),
    
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='PointSample', num_points=16384, sample_range=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='suscape_infos_train.pkl',
            data_prefix=dict(pts=''),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,            
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file='suscape_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file='suscape_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='SuscapeMetric',
    data_root='./',
    ann_file=data_root + 'suscape_infos_val.pkl')
test_evaluator = val_evaluator


# Car {'num': 491100,         'size': array([4.42419589, 1.974167  , 1.63783862])}
# Pedestrian {'num': 208470,  'size': array([0.65047053, 0.66835484, 1.6406836 ])}
# ScooterRider {'num': 75384, 'size': array([1.71321554, 0.75332566, 1.59110313])}
# Truck {'num': 84839,        'size': array([8.309413  , 2.62914506, 3.06056424])}
# Scooter {'num': 60143,      'size': array([1.64555086, 0.66646036, 1.22126398])}
# Bicycle {'num': 48172,      'size': array([1.57330043, 0.57421994, 1.14353524])}
# Van {'num': 40004,          'size': array([4.57143242, 1.98126929, 2.02480982])}
# Bus {'num': 31140,          'size': array([10.34199791,2.96619499, 3.22222376])}
# BicycleRider {'num': 18776, 'size': array([1.66885451, 0.68840352, 1.68261234])}
# Trimotorcycle {'num': 10394,'size': array([2.85369005, 1.23022373, 1.64355945])}

# FireHydrant {'num': 1828, 'size': array([0.31465264, 0.34593553, 0.79809902])}
# BicycleGroup {'num': 17036, 'size': array([1.93354299, 5.28169363, 1.26207564])}
# FreightTricycle {'num': 352, 'size': array([2.6228381 , 1.29014116, 1.59405056])}
# Cone {'num': 3982, 'size': array([0.29667343, 0.32195517, 0.69694369])}
# BabyCart {'num': 2051, 'size': array([0.89993721, 0.55076108, 1.07847937])}
# TrafficBarrier {'num': 2587, 'size': array([0.90421168, 0.92646293, 0.98071319])}
# RoadBarrel {'num': 1982, 'size': array([0.6152475 , 0.65617581, 0.8575062 ])}
# TourCar {'num': 566, 'size': array([3.7860175 , 1.69089259, 2.03223351])}
# Unknown {'num': 640, 'size': array([2.26234851, 1.26276432, 1.49466171])}
# TrashCan {'num': 92, 'size': array([3.56961823, 1.81439505, 1.94420439])}
# ConstructionCart {'num': 605, 'size': array([1.54606171, 0.82443987, 0.93345529])}
# DontCare {'num': 71, 'size': array([ 4.36643558, 16.17612857,  2.92653931])}
# PoliceCar {'num': 374, 'size': array([4.22353765, 1.9896514 , 2.04326272])}
# Animal {'num': 292, 'size': array([0.82393301, 0.33610743, 0.57356045])}
# SaftyTriangle {'num': 39, 'size': array([0.33409927, 0.68940892, 0.55627078])}
# Misc {'num': 141, 'size': array([2.76098408, 2.77951754, 1.72387715])}
# MotorcycleRider {'num': 295, 'size': array([2.1365149 , 0.872236  , 1.70562447])}
# PlatformCart {'num': 309, 'size': array([0.98497178, 0.66531463, 0.92197124])}
# Unknown1 {'num': 63, 'size': array([0.83747669, 1.12076463, 2.02212171])}
# Cart {'num': 40, 'size': array([0.35567467, 0.45984333, 0.98927057])}
# RoadRoller {'num': 78, 'size': array([3.81749284, 1.66746585, 2.45179837])}
# Unknown2 {'num': 47, 'size': array([0.75147787, 1.07919912, 2.50429424])}

# class_names = ['Car', 'Pedestrian', 'ScooterRider'
#                , 'Truck', 'Scooter',
#                 'Bicycle', 'Van', 'Bus', 'BicycleRider', #'BicycleGroup', 
#                 'Trimotorcycle', #'RoadWorker', 
#                 ]


model = dict(
    rpn_head = dict(
        num_classes=len(_base_['class_names']),        
        bbox_coder=dict(
                use_mean_size=True,
                mean_size=[ [4.42419589, 1.974167  , 1.63783862],
                            [0.65047053, 0.66835484, 1.6406836 ],
                            [1.71321554, 0.75332566, 1.59110313],
                            [8.309413  , 2.62914506, 3.06056424],
                            [1.64555086, 0.66646036, 1.22126398],
                            [1.57330043, 0.57421994, 1.14353524],
                            [4.57143242, 1.98126929, 2.02480982],
                            [10.34199791,2.96619499, 3.22222376],
                            [1.66885451, 0.68840352, 1.68261234],
                            [2.85369005, 1.23022373, 1.64355945]],
        )
    )
)


lr = 0.0005  # max learning rate
optim_wrapper = dict(optimizer=dict(lr=lr, betas=(0.95, 0.85)))

train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=1)

auto_scale_lr = dict(enable=False, base_batch_size=32)
param_scheduler = [
    # learning rate scheduler
    # During the first 35 epochs, learning rate increases from 0 to lr * 10
    # during the next 45 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=35,
        eta_min=lr * 10,
        begin=0,
        end=35,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        eta_min=lr * 1e-2,
        begin=35,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 35 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 45 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=35,
        eta_min=0.85 / 0.95,
        begin=0,
        end=35,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=45,
        eta_min=1,
        begin=35,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True)
]
