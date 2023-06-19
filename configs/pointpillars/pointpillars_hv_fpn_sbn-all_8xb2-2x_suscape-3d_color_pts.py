_base_ = [
    '../_base_/models/pointpillars_hv_fpn_suscape_color_pts.py',
    '../_base_/datasets/suscape-3d-color-pts.py', '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).

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



auto_scale_lr = dict(enable=False, base_batch_size=16)
