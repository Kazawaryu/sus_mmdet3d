_base_ = [
    '../_base_/models/point_rcnn_2x8.py',
    "../_base_/datasets/suscape-3d.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cyclic-20e.py"
]

dataset_type = "SuscapeDataset"
data_root = "data/suscape/"
class_names = [
    "Car",
    "Pedestrian",
    "ScooterRider",
    "Truck",
    "Scooter",
    "Bicycle",
    "Van",
    "Bus",
    "BicycleRider",
    "Trimotorcycle",
]

point_cloud_range = [-80, -80, -5, 80, 80, 3]

model = dict(
    rpn_head=dict(
        num_classes=10,
        bbox_coder=dict(
            use_mean_size=True,
            mean_size=[
                
            ]
 	    )
    )
)


