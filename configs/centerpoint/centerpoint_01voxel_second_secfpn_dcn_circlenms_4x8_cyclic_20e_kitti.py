_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_kitti.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))