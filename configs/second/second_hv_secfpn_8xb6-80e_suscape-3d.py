_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/suscape-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
)


test_dataloader = dict(
    batch_size=2,
    num_workers=2,
)



lr = 0.0005
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the official AdamW optimizer implemented by PyTorch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=lr * 10,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=lr * 1e-4,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=16,
        eta_min=0.85 / 0.95,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=24,
        eta_min=1,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]

