# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pandas as pd
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from pyquaternion import Quaternion

from mmdet3d.evaluation import suscape_eval
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SuscapeMetric(BaseMetric):
    """Suscape evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        csv_savepath (str, optional): The path for saving csv files. It
            includes the file path and the csv filename, e.g.,
            "a/b/filename.csv". If not specified, the result will not be
            converted to csv file. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality=dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                 prefix: Optional[str] = None,
                 jsonfile_prefix: str = None,
                 format_only: bool = False,
                 csv_savepath: str = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Suscape metric'
        super(SuscapeMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.jsonfile_prefix = jsonfile_prefix
        self.format_only = format_only
        if self.format_only:
            assert csv_savepath is not None, 'csv_savepath must be not None '
            'when format_only is True, otherwise the result files will be '
            'saved to a temp directory which will be cleaned up at the end.'

        self.backend_args = backend_args
        self.csv_savepath = csv_savepath
        self.metrics = metric if isinstance(metric, list) else [metric]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def format_results_suscape(self,
                       outputs,
                       score_threshold=0.0,
                       submission_prefix=None):
        assert isinstance(outputs, list), 'results must be a list'
        

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_suscape(outputs, self.dataset_meta['classes'],score_threshold,
                                                    submission_prefix)
        elif 'pred_instances_3d' in outputs[0]:
            results_ = [out["pred_instances_3d"] for out in outputs]
            result_files = self.bbox2result_suscape(
                        results_, self.dataset_meta['classes'], score_threshold,
                        submission_prefix)          
        else:
            result_files = self.bbox2result_suscape(outputs, self.dataset_meta['classes'],score_threshold,
                                                  submission_prefix)
        return result_files

    def bbox2result_suscape(self,
                          net_outputs,
                          class_names,
                          score_threshold=0,
                          submission_prefix=None):
        assert len(net_outputs) == len(self.data_infos['data_list']), \
            'invalid list length of network outputs'

        det_annos = []
        print('\nConverting prediction to suscape format')
        for idx, pred_dicts in enumerate(net_outputs):

            info = self.data_infos['data_list'][idx]
            annos = {
                'scene': info['scene_token'],
                'frame': info['frame_token'],
                'objs': []
                }
            
            #sample_idx = info['image']['image_idx']
            #image_shape = info['image']['image_shape'][:2]

            # box_dict = self.convert_valid_bboxes(pred_dicts["boxes_3d"],pred_dicts["scores_3d"], pred_dicts["labels_3d"])
            if len(pred_dicts['scores_3d']) != 0:
                for tensor, score, label in zip(pred_dicts['bboxes_3d'].tensor.numpy(), pred_dicts['scores_3d'].numpy(), pred_dicts['labels_3d'].numpy()):
                    
                    if score < score_threshold:
                        continue
                    
                    obj = {
                        'psr':{
                            'position': {
                                'x': tensor[0],
                                'y': tensor[1],
                                'z': tensor[2] + tensor[5]/2
                            },
                            'scale': {
                                'x': tensor[3],
                                'y': tensor[4],
                                'z': tensor[5]
                            },
                            'rotation': {
                                'x': 0,
                                'y': 0,
                                'z': tensor[6]
                            }
                        },
                        'score': score,
                        'obj_type': class_names[label],
                        'obj_id': '',
                        'obj_attr': str(score)
                    }

                    annos['objs'].append(obj)

            det_annos.append(annos)

        
        if submission_prefix is not None:
            # submission_file = submission_prefix + '.json'
            # print(f'\nConverting prediction to {submission_file}')
            # mmcv.dump(det_annos, submission_file)
            # print(f'Result is saved to {submission_file}.')

            # mmengine.mkdir_or_exist(submission_prefix)
            print(f'save prediction to {submission_prefix}')
            for d in det_annos:
                os.makedirs(submission_prefix + '/' + d['scene'], exist_ok=True)
                path = submission_prefix + '/' + d['scene'] + '/label/' + d['frame'] + '.json'

                mmengine.dump(d, path)
        
        return det_annos

    def format_gt_to_suscape(self, gt):
        """Format suscape gt to suscape format.      
        """
        print('\nConverting ground truth to suscape format')

      
        gt_annos = []
        
        
        for  frame in gt:
            ann = {
                'objs':  [],
                'scene': frame['scene_token'],
                'frame': frame['frame_token'],
            }
            for inst in frame['instances']:
                obj = {
                    'psr':{
                        'position':{
                            'x': inst['bbox_3d'][0],
                            'y': inst['bbox_3d'][1],
                            'z': inst['bbox_3d'][2],
                        },
                        'scale':{
                            'x': inst['bbox_3d'][3],
                            'y': inst['bbox_3d'][4],
                            'z': inst['bbox_3d'][5],
                        },
                        'rotation':{
                            'x': 0,
                            'y': 0,
                            'z': inst['bbox_3d'][6],
                        },
                    },
                    'obj_type': inst['class'],
                    
                }

                ann['objs'].append(obj)
            gt_annos.append(ann)

        return gt_annos
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']

        # load annotations
        self.data_infos = load(
            osp.join(self.data_root, self.ann_file),
            backend_args=self.backend_args)
        

        # suscape metrics
        print(' compute suscape metrics')
        result_files = self.format_results_suscape(results, 
                                                   score_threshold=0.0, 
                                                   submission_prefix=self.jsonfile_prefix)
        from suscape.eval.detection_3d import evaluate as suscape_evaluate
        import json
        
        gt_annos = self.format_gt_to_suscape(self.data_infos['data_list'])
    
        dt_res = result_files

        metrics, metrics_string = suscape_evaluate(dt_res, gt_annos, self.dataset_meta['classes'])
        print(metrics_string)


        # lyft metrics
        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix,
                                                   self.csv_savepath)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.csv_savepath)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self.lyft_evaluate(
                result_dict, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def format_results(
        self,
        results: List[dict],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None,
        csv_savepath: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            csv_savepath (str, optional): The path for saving csv files. It
                includes the file path and the csv filename, e.g.,
                "a/b/filename.csv". If not specified, the result will not be
                converted to csv file. Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if 'pred' in name and '3d' in name and name[0] != '_':
                print(f'\nFormating bboxes of {name}')
                # format result of model output in Det3dDataSample,
                # include 'pred_instances_3d','pts_pred_instances_3d',
                # 'img_pred_instances_3d'
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_dict[name] = self._format_bbox(results_,
                                                      sample_idx_list, classes,
                                                      tmp_file_)
        if csv_savepath is not None:
            if 'pred_instances_3d' in result_dict:
                self.json2csv(result_dict['pred_instances_3d'], csv_savepath)
            elif 'pts_pred_instances_3d' in result_dict:
                self.json2csv(result_dict['pts_pred_instances_3d'],
                              csv_savepath)
        return result_dict, tmp_dir

    def json2csv(self, json_path: str, csv_savepath: str) -> None:
        """Convert the json file to csv format for submission.

        Args:
            json_path (str): Path of the result json file.
            csv_savepath (str): Path to save the csv file.
        """
        results = mmengine.load(json_path)['results']
        sample_list_path = osp.join(self.data_root, 'sample_submission.csv')
        data = pd.read_csv(sample_list_path)
        Id_list = list(data['Id'])
        pred_list = list(data['PredictionString'])
        cnt = 0
        print('Converting the json to csv...')
        for token in results.keys():
            cnt += 1
            predictions = results[token]
            prediction_str = ''
            for i in range(len(predictions)):
                prediction_str += \
                    str(predictions[i]['score']) + ' ' + \
                    str(predictions[i]['translation'][0]) + ' ' + \
                    str(predictions[i]['translation'][1]) + ' ' + \
                    str(predictions[i]['translation'][2]) + ' ' + \
                    str(predictions[i]['size'][0]) + ' ' + \
                    str(predictions[i]['size'][1]) + ' ' + \
                    str(predictions[i]['size'][2]) + ' ' + \
                    str(Quaternion(list(predictions[i]['rotation']))
                        .yaw_pitch_roll[0]) + ' ' + \
                    predictions[i]['name'] + ' '
            prediction_str = prediction_str[:-1]
            idx = Id_list.index(token)
            pred_list[idx] = prediction_str
        df = pd.DataFrame({'Id': Id_list, 'PredictionString': pred_list})
        mmengine.mkdir_or_exist(os.path.dirname(csv_savepath))
        df.to_csv(csv_savepath, index=False)

    def _format_bbox(self,
                     results: List[dict],
                     sample_idx_list: List[int],
                     classes: Optional[List[str]] = None,
                     jsonfile_prefix: Optional[str] = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        lyft_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes = output_to_lyft_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos['data_list'][sample_idx]['frame_path']
            # boxes = lidar_lyft_box_to_global(self.data_infos[sample_idx],
            #                                  boxes)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                lyft_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    name=name,
                    score=box.score)
                annos.append(lyft_anno)
            lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos,
        }

        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmengine.dump(lyft_submissions, res_path)
        return res_path

    def lyft_evaluate(self,
                      result_dict: dict,
                      metric: str = 'bbox',
                      logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in Lyft protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated. Defaults to 'bbox'.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Evaluation results.
        """
        metric_dict = dict()
        for name in result_dict:
            print(f'Evaluating bboxes of {name}')
            ret_dict = self._evaluate_single(
                result_dict[name], logger=logger, result_name=name)
            metric_dict.update(ret_dict)
        return metric_dict


    def load_suscape_predictions(self, res_path):
        """Load Lyft predictions from json file.

        Args:
            res_path (str): Path of result json file recording detections.

        Returns:
            list[dict]: List of prediction dictionaries.
        """
        predictions = mmengine.load(res_path)
        predictions = predictions['results']
        all_preds = []
        for sample_token in predictions.keys():
            all_preds.extend(predictions[sample_token])
        return all_preds

    def format_gts_lyft_format(self):
        gts = []
        for info in self.data_infos['data_list']:
            sample_token = info['frame_path']
            for obj in info['instances']:
                gts.append({
                    'sample_token': sample_token,
                    'translation': obj['bbox_3d'][0:3],
                    'rotation': [np.cos(obj['bbox_3d'][6]/2), 0, 0, np.sin(obj['bbox_3d'][6]/2)], #quaternion, w,x,y,z (Hamilton format)
                    'size': [obj['bbox_3d'][4], obj['bbox_3d'][3],obj['bbox_3d'][5]], #'wlh'
                    'name': self.dataset_meta['classes'][obj['bbox_label_3d']],
                })

        return gts
    
    def _evaluate_single(self,
                         result_path: str,
                         logger: MMLogger = None,
                         result_name: str = 'pts_bbox') -> dict:
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pts_bbox'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        # lyft = Lyft(
        #     data_path=osp.join(self.data_root, self.version),
        #     json_path=osp.join(self.data_root, self.version, self.version),
        #     verbose=True)
        # eval_set_map = {
        #     'v1.01-train': 'val',
        # }


        # gts, predictions, class_names, 
        gts = self.format_gts_lyft_format()
        predictions = self.load_suscape_predictions(result_path)
        metrics = suscape_eval(gts, predictions, self.dataset_meta['classes'], 
                             output_dir, logger)

        # record metrics
        detail = dict()
        metric_prefix = f'{result_name}_Lyft'

        for i, name in enumerate(metrics['class_names']):
            AP = float(metrics['mAPs_cate'][i])
            detail[f'{metric_prefix}/{name}_AP'] = AP

        detail[f'{metric_prefix}/mAP'] = metrics['Final mAP']
        return detail


def output_to_lyft_box(detection: dict) -> List[LyftBox]:
    """Convert the output to the box class in the Lyft.

    Args:
        detection (dict): Detection results.

    Returns:
        List[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    # our LiDAR coordinate system -> Lyft box coordinate system
    lyft_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(bbox3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        box = LyftBox(
            box_gravity_center[i],
            lyft_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i])
        box_list.append(box)
    return box_list


