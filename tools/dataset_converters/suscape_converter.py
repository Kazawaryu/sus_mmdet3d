# Copyright (c) OpenMMLab. All rights reserved.
import os
from logging import warning
from os import path as osp
import mmengine
import numpy as np
import json
from .suscape_dataset import SuscapeDataset


## pcd parser

import re
import warnings


class BinPcdReader:
    """ Read binary PCD files.
    """
    def __init__(self, filename):
        self.filename = filename
        self.metadata = None
        self.points = None


        numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
        #numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
        self.pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


        self._read()

    def _read(self):
        with open(self.filename, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip()
                header.append(ln)
                if ln.startswith('DATA'.encode()):
                    break

            metadata = self.parse_header(header)
            dtype = self._build_dtype(metadata)
            self.pc_data = self.parse_binary_pc_data(f, dtype, metadata)

  
    def parse_binary_pc_data(self, f, dtype, metadata):
        # print("the dtype.itemsize is: ",dtype.itemsize)
        # print("the dtype['x'] is: ",dtype['x'])
        rowstep = metadata['points']*dtype.itemsize
        # for some reason pcl adds empty space at the end of files
        buf = f.read(rowstep)
        return np.fromstring(buf, dtype=dtype)
    
    def parse_header(self, lines):
        """ Parse header of PCD files.
        """
        metadata = {}
        for ln in lines:
            if ln.startswith('#'.encode()) or len(ln) < 2:
                continue
            match = re.match('(\w+)\s+([\w\s\.]+)', ln.decode())
            if not match:
                warnings.warn("warning: can't understand line: %s" % ln)
                continue
            key, value = match.group(1).lower(), match.group(2)
            if key == 'version':
                metadata[key] = value
            elif key in ('fields', 'type'):
                metadata[key] = value.split()
            elif key in ('size', 'count'):
                metadata[key] = [int(i) for i in value.split()]
                # metadata[key] = map(int, value.split())
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(value)
            elif key == 'viewpoint':
                metadata[key] = map(float, value.split())
            elif key == 'data':
                metadata[key] = value.strip().lower()
            # TODO apparently count is not required?
        # add some reasonable defaults
        if 'count' not in metadata:
            metadata['count'] = [1]*len(metadata['fields'])
        if 'viewpoint' not in metadata:
            metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if 'version' not in metadata:
            metadata['version'] = '.7'
        return metadata

    def _metadata_is_consistent(metadata):
        """ Sanity check for metadata. Just some basic checks.
        """
        checks = []
        required = ('version', 'fields', 'size', 'width', 'height', 'points',
                    'viewpoint', 'data')
        for f in required:
            if f not in metadata:
                print('%s required' % f)
        checks.append((lambda m: all([k in m for k in required]),
                    'missing field'))
        # print("te len of the list(m['count']) is: ",list(m['count']))
        checks.append((lambda m: len(m['type']) == len(list(m['count'])) ==
                    len(list(m['fields'])),
                    'length of type, count and fields must be equal'))
        checks.append((lambda m: m['height'] > 0,
                    'height must be greater than 0'))
        checks.append((lambda m: m['width'] > 0,
                    'width must be greater than 0'))
        checks.append((lambda m: m['points'] > 0,
                    'points must be greater than 0'))
        checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                    'binary_compressed'),
                    'unknown data type:'
                    'should be ascii/binary/binary_compressed'))
        ok = True
        for check, msg in checks:
            if not check(metadata):
                print('error:', msg)
                ok = False
        return ok

    def _build_dtype(self, metadata):
        """ Build numpy structured array dtype from pcl metadata.

        Note that fields with count > 1 are 'flattened' by creating multiple
        single-count fields.

        *TODO* allow 'proper' multi-count fields.
        """
        fieldnames = []
        typenames = []
        other = 1
        for f, c, t, s in zip(metadata['fields'],
                            metadata['count'],
                            metadata['type'],
                            metadata['size']):
            np_type = self.pcd_type_to_numpy_type[(t, s)]
            if c == 1:
                fieldnames.append(f)
                typenames.append(np_type)
            else:
                fieldnames.extend(['%s_%s_%04d' % (f,str(other), i) for i in range(c)])
                typenames.extend([np_type]*c)
                other = other + 1
        a = list(zip(fieldnames, typenames))
        dtype = np.dtype(list(zip(fieldnames, typenames)))
        return dtype


####################################################################3


def _read_list_from_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



METAINFO = {
    'classes': ['Car', 'Pedestrian', 'ScooterRider', 'Truck', 'Scooter',
                'Bicycle', 'Van', 'Bus', 'BicycleRider', #'BicycleGroup', 
                'Trimotorcycle', #'RoadWorker', 
                
                # 'LongVehicle', 'Cone', 
                # 'TrafficBarrier', 'ConcreteTruck', 'Child', 'BabyCart', 
                # 'RoadBarrel', #'FireHydrant', 
                # #'MotorcyleRider', 
                # 'Crane', 
                # 'ForkLift', 'Bulldozer', 'Excavator', 
                #'Motorcycle'
                ],     
    'classMap': {
        'Motorcycle': 'Scooter',
        'MotorcyleRider': 'ScooterRider',
        'RoadWorker': 'Pedestrian',
        'LongVehicle': 'Truck',
        'ConcreteTruck': 'Truck',
        'Child': 'Pedestrian',
        'Crane': 'Truck',
        'ForkLift': 'Truck',
        'Bulldozer': 'Truck', 
        'Excavator': 'Truck',
    }
}


def create_suscape_infos(root_path,
                      info_prefix,
                      out_path,
                      version='v1.0-train',
                      max_sweeps=1):
    """Create info file of suscape dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-train'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 1.
    """

    available_vers = ['v1.0-train', 'v1.0-test']
    assert version in available_vers
    if version == 'v1.0-train':
        train_scenes = _read_list_from_file('data/suscape/train.txt')
        val_scenes = _read_list_from_file('data/suscape/val.txt')
    elif version == 'v1.0-test':
        train_scenes = _read_list_from_file('data/suscape/test.txt')
        val_scenes = []
    else:
        raise ValueError('unknown')

    suscape = SuscapeDataset(root_path)
    # filter existing scenes.
    available_scenes = suscape.get_scene_names()
    

    available_scene_names = available_scenes
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, \
                val scene: {len(val_scenes)}')
        
    train_infos, val_infos = _fill_trainval_infos(
        suscape, train_scenes, val_scenes, out_path, test, max_sweeps=max_sweeps)

    metainfo = dict(version=version, dataset='suscape')
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    metainfo['version'] = version
    metainfo['info_version'] = '1.1'

    if test:
        print(f'test sample: {len(train_infos)}')
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_name = f'{info_prefix}_infos_test'
        info_path = osp.join(out_path, f'{info_name}.pkl')
        mmengine.dump(data, info_path)
    else:
        print(f'train sample: {len(train_infos)}, \
                val sample: {len(val_infos)}')
        data = dict(data_list=train_infos, metainfo=metainfo)
        train_info_name = f'{info_prefix}_infos_train'
        info_path = osp.join(out_path, f'{train_info_name}.pkl')
        mmengine.dump(data, info_path)
        data = dict(data_list=val_infos, metainfo=metainfo)
        val_info_name = f'{info_prefix}_infos_val'
        info_val_path = osp.join(out_path, f'{val_info_name}.pkl')
        mmengine.dump(data, info_val_path)

def _read_image_info(root_path, scene, frame, camera_type, camera):
    with open(os.path.join(root_path, scene, 'calib', camera_type, camera+'.json')) as f:
        calib = json.load(f)

        return {
            'lidar2cam': np.reshape(np.array(calib['extrinsic']),(4,4)),
            'cam2img': np.reshape(np.array(calib['intrinsic']), (3,3)),     
            'height': 1536,
            'width': 2048,      
        }


def _read_scene(suscape, out_path, scene_name, overwrite_lidar_file=False):
    
    scene = suscape.get_scene_info(scene_name)

    infos = []
    for frame in scene['frames']:
        
        # write lidar bin files
        bin_lidar_path = osp.join(out_path, "lidar.bin", scene_name, "lidar")
        if os.path.exists(bin_lidar_path) == False:
            os.makedirs(bin_lidar_path)

        bin_lidar_file = osp.join(bin_lidar_path, frame+".bin")

        if not os.path.exists(bin_lidar_file) or overwrite_lidar_file :
            pcd_reader = BinPcdReader(osp.join(suscape.cfg['lidar_dir'], scene_name, 'lidar', frame+scene['lidar_ext']))
            
            bin_data = np.stack([pcd_reader.pc_data[x] for x in ['x', 'y', 'z', 'intensity']], axis=-1)
            bin_data = bin_data[(bin_data[:,0]!=0) & (bin_data[:,1]!=0) & (bin_data[:,2]!=0)]
            bin_data.tofile(bin_lidar_file)

        
        info = {
            'frame_path': scene_name+'/'+frame,
            'lidar_points': {
                'lidar_path': os.path.join("lidar.bin", scene_name, "lidar", frame+".bin"),
                'num_pts_feats': 4,
                'lidar2ego': np.eye(4),
                },
             'images': {},
             'ego2global': scene['lidar_pose'][frame]['lidarPose'],
             'sweeps': [],
            # 'cams': dict(),
            # 'lidar2ego_translation': cs_record['translation'],
            # 'lidar2ego_rotation': cs_record['rotation'],
            # 'ego2global_translation': pose_record['translation'],
            # 'ego2global_rotation': pose_record['rotation'],
            # 'timestamp': sample['timestamp'],
        }

        # info['images']['camera/front'] = _read_image_info(root_path, scene, frame, 'camera', 'front')

       
        for cam in scene['camera']:
            lidar2cam = suscape.get_calib_lidar2cam(scene, frame, 'camera', cam)
            info['images'][cam] = {
                'img_path': os.path.join(suscape.camera_dir, scene_name, 'camera', cam, frame+scene['camera_ext']),
                'cam2img': scene['calib']['camera'][cam]['intrinsic'],
                'cam2ego': np.linalg.inv(lidar2cam).tolist(),
                'lidar2cam': lidar2cam.tolist()
            }


        objs = suscape.read_label(scene_name, frame)['objs']
        if len(objs) == 0:
            print("no objs: ", osp.join(scene_name, 'label', frame+".json"))
            continue

        info['instances'] = []
        for o in objs:
            
            if not o['obj_type'] in METAINFO['classes']:
                # inst['bbox_label'] = -1
                # skip this obj
                continue
            obj_type = METAINFO['classMap'][o['obj_type']] if o['obj_type'] in METAINFO['classMap'] else o['obj_type']
            inst = dict()
            inst['class'] = obj_type
            inst['bbox_label'] = METAINFO['classes'].index(obj_type)

            inst['bbox_3d'] = [o['psr']['position']['x'],
                                          o['psr']['position']['y'],
                                          o['psr']['position']['z'],
                                          o['psr']['scale']['x'],
                                          o['psr']['scale']['y'],
                                          o['psr']['scale']['z'],
                                          o['psr']['rotation']['z'],
                                          ]

            inst['truncated'] = 0
            inst['alpha'] = 0
            inst['occluded'] = 0
            inst['bbox'] = [0,0,0,0]
            inst['score'] = 0
            inst['bbox_label_3d'] = inst['bbox_label']
            inst['truncated'] = 0
            inst['alpha'] = 0
            inst['occluded'] = 0
            inst['bbox'] = [0,0,0,0]

            info['instances'].append(inst)

        infos.append(info)
    return infos

def _fill_trainval_infos(suscape,
                         train_scenes,
                         val_scenes,
                         out_path,
                         test=False,
                         max_sweeps=1):
    """Generate the train/val infos from the raw data.

    Args:
        root_path: suscape dataset root path
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 1.

    Returns:
        tuple[list[dict]]: Information of training set and
            validation set that will be saved to the info file.
    """
    train_infos = []
    val_infos = []

    for scene in train_scenes:
        print(scene)
        infos = _read_scene(suscape, out_path, scene)
        train_infos.extend(infos)
    for scene in val_scenes:
        print(scene)
        infos = _read_scene(suscape, out_path, scene)
        val_infos.extend(infos)
    return train_infos, val_infos

