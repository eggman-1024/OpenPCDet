import os
import pickle
import copy
import torch
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, calibration_kitti, box_utils
from .kitti2waymo_utils import *
from .waymo_dataset import WaymoDataset


class Kitti2WaymoDataset(WaymoDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)


    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        # 处理每一段数据
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        # 原waymo在train时会每5帧采样一次，但这里不需要
        # if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
        #     sampled_waymo_infos = []
        #     for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
        #         sampled_waymo_infos.append(self.infos[k])
        #     self.infos = sampled_waymo_infos
        #     self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
            
        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def get_infos(self, segment_info_path, kitti_path, save_path, 
                  num_workers=multiprocessing.cpu_count(), has_label=True, 
                  sampled_interval=1, update_info_only=False):
        from . import kitti2waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            kitti2waymo_utils.process_single_sequence,
            kitti_path=kitti_path, save_path=save_path, sampled_interval=sampled_interval, 
            carla_data=True, has_label=has_label, update_info_only=update_info_only
        )

        sample_sequence_info_list = [read_pickle(segment_info_path / (sequence_file + '.pkl')) for sequence_file in self.sample_sequence_list]
        # process_single_sequence(sample_sequence_file_list[0])
        DEBUG = False
        if DEBUG:
            sequence_infos = []
            for cnt, sequence_file in enumerate(sample_sequence_info_list):
                print('Processing %s' % sequence_file)
                sequence_info = process_single_sequence(sequence_file)
                sequence_infos.append(sequence_info)
        else:
            with multiprocessing.Pool(num_workers) as p:
                sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_info_list),
                                        total=len(sample_sequence_info_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos


    def get_lidar(self, sequence_name, sample_idx):
        """
        Get lidar data by index
        :param idx: index of the lidar data
        :return: lidar data
        """
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        points = np.load(lidar_file)  
        # (N, 12): ['x', 'y', 'z', 'cos_angle', 'vr', 'vx', 'vy', 'vz', 'v_cps', 'intensity', 'id', 'label']

        return points
    
    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, load_pred_boxes=False):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        # TODO: 若load_pred_boxes=True, 这里需要修改
        def load_pred_boxes_from_dict(sequence_name, sample_idx):
            """
            boxes: (N, 11)  [x, y, z, dx, dy, dn, raw, vx, vy, score, label]
            """
            sequence_name = sequence_name.replace('training_', '').replace('validation_', '')
            load_boxes = self.pred_boxes_dict[sequence_name][sample_idx]
            assert load_boxes.shape[-1] == 11
            load_boxes[:, 7:9] = -0.1 * load_boxes[:, 7:9]  # transfer speed to negtive motion from t to t-1
            return load_boxes
        
        pose_cur = info['pose'].reshape((4, 4))
        num_pts_cur = points.shape[0]
        sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1]), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]

        if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
            onehot_cur = np.zeros((points.shape[0], len(sample_idx_pre_list) + 1)).astype(points.dtype)
            onehot_cur[:, 0] = 1
            points = np.hstack([points, onehot_cur])
        else:
            points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        points_pre_all = []
        num_points_pre = []

        pose_all = [pose_cur]
        pred_boxes_all = []
        if load_pred_boxes:
            pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx)
            pred_boxes_all.append(pred_boxes)

        sequence_info = self.seq_name_to_infos[sequence_name]

        # 遍历历史帧索引
        for idx, sample_idx_pre in enumerate(sample_idx_pre_list):

            # 加载点云数据和位姿信息
            points_pre = self.get_lidar(sequence_name, sample_idx_pre)
            pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1) # 转换为齐次矩阵
            # 从局部坐标系转换到全局坐标系
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            # 从全局坐标系转换到当前帧的局部坐标系
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            # 拼接其他特征
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
                onehot_vector = np.zeros((points_pre.shape[0], len(sample_idx_pre_list) + 1))
                onehot_vector[:, idx + 1] = 1
                points_pre = np.hstack([points_pre, onehot_vector])
            else:
                # add timestamp
                points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            points_pre = remove_ego_points(points_pre, 1.0)
            points_pre_all.append(points_pre)
            num_points_pre.append(points_pre.shape[0])
            pose_all.append(pose_pre)

            if load_pred_boxes:
                pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
                pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx_pre)
                pred_boxes = self.transform_prebox_to_current(pred_boxes, pose_pre, pose_cur)
                pred_boxes_all.append(pred_boxes)

        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        num_points_all = np.array([num_pts_cur] + num_points_pre).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        if load_pred_boxes:
            temp_pred_boxes = self.reorder_rois_for_refining(pred_boxes_all)
            pred_boxes = temp_pred_boxes[:, :, 0:9]
            pred_scores = temp_pred_boxes[:, :, 9]
            pred_labels = temp_pred_boxes[:, :, 10]
        else:
            pred_boxes = pred_scores = pred_labels = None

        return points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    # 不支持shared memory
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }

        points = self.get_lidar(sequence_name, sample_idx)

        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:
            points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.dataset_cfg.get('USE_PREDBOX', False)
            )
            input_dict['poses'] = poses
            if self.dataset_cfg.get('USE_PREDBOX', False):
                input_dict.update({
                    'roi_boxes': pred_boxes,
                    'roi_scores': pred_scores,
                    'roi_labels': pred_labels,
                })

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            # 过滤空边界框
            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])  # 优先使用 info['metadata']，如果不存在，则使用 info['frame_id'] 作为默认值
        data_dict.pop('num_points_in_gt', None)
        return data_dict


    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=1,
                                    processed_data_tag=None, carla_data=True):
        
        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED

        if use_sequence_data:
            st_frame, ed_frame = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0], self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[1]
            self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0] = min(-4, st_frame)  # at least we use 5 frames for generating gt database to support various sequence configs (<= 5 frames)
            st_frame = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0]
            database_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
            db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d_multiframe_%s_to_%s.pkl' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
            db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s_global.npy' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame))
        else:
            database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
            db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
            db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in tqdm(range(0, len(infos), sampled_interval)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            if use_sequence_data:
                points, num_points_all, sample_idx_pre_list, _, _, _, _ = self.get_sequence_data(
                    info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG
                )

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            """
            按帧索引过滤目标类别：
            每隔 4 帧，移除 Vehicle 类别的目标。
            每隔 2 帧，移除 Pedestrian 类别的目标
            waymo数据集很大, 因此需要对数据进行采样, 但是我们的carla数据集很小, 因此不需要采样
            """
            if not carla_data:
                if k % 4 != 0 and len(names) > 0:
                    mask = (names == 'Vehicle')
                    names = names[~mask]
                    difficulty = difficulty[~mask]
                    gt_boxes = gt_boxes[~mask]

                if k % 2 != 0 and len(names) > 0:
                    mask = (names == 'Pedestrian')
                    names = names[~mask]
                    difficulty = difficulty[~mask]
                    gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue
            
            # 判断点云中的每个点是否位于给定的 3D 边界框内，并返回每个点所属的边界框索引(不在任何边界框内的点索引为 -1)
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            # 遍历每个gt物体
            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    gt_points = gt_points.astype(np.float32)
                    assert gt_points.dtype == np.float32
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        # pickle文件按name类别保存所有的gt信息(e.g. Vehicle, Pedestrian, Cyclist)
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)

    def create_gt_database_of_single_scene(self, info_with_idx, database_save_path=None, use_sequence_data=False, used_classes=None,
                                           total_samples=0, use_cuda=False, crop_gt_with_tail=False, carla_data=True):
        info, info_idx = info_with_idx
        print('gt_database sample: %d/%d' % (info_idx, total_samples))

        all_db_infos = {}
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = self.get_lidar(sequence_name, sample_idx)

        if use_sequence_data:
            points, num_points_all, sample_idx_pre_list, _, _, _, _ = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG
            )

        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        gt_boxes = annos['gt_boxes_lidar']

        if not carla_data:
            if info_idx % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if info_idx % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            return {}

        if use_sequence_data and crop_gt_with_tail:
            assert gt_boxes.shape[1] == 9
            speed = gt_boxes[:, 7:9]
            sequence_cfg = self.dataset_cfg.SEQUENCE_CONFIG
            assert sequence_cfg.SAMPLE_OFFSET[1] == 0
            assert sequence_cfg.SAMPLE_OFFSET[0] < 0
            num_frames = sequence_cfg.SAMPLE_OFFSET[1] - sequence_cfg.SAMPLE_OFFSET[0] + 1
            assert num_frames > 1
            latest_center = gt_boxes[:, 0:2]
            oldest_center = latest_center - speed * (num_frames - 1) * 0.1
            new_center = (latest_center + oldest_center) * 0.5
            new_length = gt_boxes[:, 3] + np.linalg.norm(latest_center - oldest_center, axis=-1)
            gt_boxes_crop = gt_boxes.copy()
            gt_boxes_crop[:, 0:2] = new_center
            gt_boxes_crop[:, 3] = new_length

        else:
            gt_boxes_crop = gt_boxes

        if use_cuda:
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes_crop[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
        else:
            box_point_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]).float(),
                torch.from_numpy(gt_boxes_crop[:, 0:7]).float()
            ).long().numpy()  # (num_boxes, num_points)

        for i in range(num_obj):
            filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
            filepath = database_save_path / filename
            if use_cuda:
                gt_points = points[box_idxs_of_pts == i]
            else:
                gt_points = points[box_point_mask[i] > 0]

            gt_points[:, :3] -= gt_boxes[i, :3]

            if (used_classes is None) or names[i] in used_classes:
                gt_points = gt_points.astype(np.float32)
                assert gt_points.dtype == np.float32
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                            'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                            'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i],
                            'box3d_crop': gt_boxes_crop[i]}

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        return all_db_infos

    def create_groundtruth_database_parallel(self, info_path, save_path, used_classes=None, split='train', sampled_interval=1,
                                             processed_data_tag=None, num_workers=16, crop_gt_with_tail=False, carla_data=True):
        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if use_sequence_data:
            st_frame, ed_frame = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0], self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[1]
            self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0] = min(-4, st_frame)  # at least we use 5 frames for generating gt database to support various sequence configs (<= 5 frames)
            st_frame = self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0]
            database_save_path = save_path / ('%s_gt_database_%s_sampled_%d_multiframe_%s_to_%s_%sparallel' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame, 'tail_' if crop_gt_with_tail else ''))
            db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d_multiframe_%s_to_%s_%sparallel.pkl' % (processed_data_tag, split, sampled_interval, st_frame, ed_frame, 'tail_' if crop_gt_with_tail else ''))
        else:
            database_save_path = save_path / ('%s_gt_database_%s_sampled_%d_parallel' % (processed_data_tag, split, sampled_interval))
            db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d_parallel.pkl' % (processed_data_tag, split, sampled_interval))
            
        database_save_path.mkdir(parents=True, exist_ok=True)

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        print(f'Number workers: {num_workers}')
        create_gt_database_of_single_scene = partial(
            self.create_gt_database_of_single_scene,
            use_sequence_data=use_sequence_data, database_save_path=database_save_path,
            used_classes=used_classes, total_samples=len(infos), use_cuda=False,
            crop_gt_with_tail=crop_gt_with_tail, carla_data=carla_data
        )
        # create_gt_database_of_single_scene((infos[300], 0))
        with multiprocessing.Pool(num_workers) as p:
            all_db_infos_list = list(p.map(create_gt_database_of_single_scene, zip(infos, np.arange(len(infos)))))

        all_db_infos = {}

        for cur_db_infos in all_db_infos_list:
            for key, val in cur_db_infos.items():
                if key not in all_db_infos:
                    all_db_infos[key] = val
                else:
                    all_db_infos[key].extend(val)

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_waymo_infos(dataset_cfg, class_names, kitti_path, data_path, save_path,
                       segment_info_tag='segment_info', processed_data_tag='waymo_processed_data',
                       workers=min(16, multiprocessing.cpu_count()), update_info_only=False,
                       carla_data=True):
    dataset = Kitti2WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 禁用GPU，因为数据预处理是I/O密集型任务，不需要GPU
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        segment_info_path=data_path / segment_info_tag,
        kitti_path=kitti_path,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, update_info_only=update_info_only
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        segment_info_path=data_path / segment_info_tag,
        kitti_path=kitti_path,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, update_info_only=update_info_only
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    if update_info_only:
        return

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag,
        carla_data=carla_data
    )
    print('---------------Data preparation Done---------------')


def create_waymo_gt_database(
    dataset_cfg, class_names, data_path, save_path, processed_data_tag='waymo_processed_data',
    workers=min(16, multiprocessing.cpu_count()), use_parallel=False, crop_gt_with_tail=False, carla_data=True):
    dataset = Kitti2WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split = 'train'
    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)

    if use_parallel:
        dataset.create_groundtruth_database_parallel(
            info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
            used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag,
            num_workers=workers, crop_gt_with_tail=crop_gt_with_tail
        )
    else:
        dataset.create_groundtruth_database(
            info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
            used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag,
            carla_data=carla_data
        )
    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    parser.add_argument('--update_info_only', action='store_true', default=False, help='')  # 是否只更新数据集信息文件
    parser.add_argument('--use_parallel', action='store_true', default=False, help='')
    parser.add_argument('--wo_crop_gt_with_tail', action='store_true', default=False, help='')
    parser.add_argument('--carla_data', default=True, help='if the data is from carla')
    parser.add_argument('--sequence_length', type=int, default=80, help='the length of sequence')

    args = parser.parse_args()

    # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR = Path('/home/public/zrtdata')
    KITTI_PATH = Path('/home/public/zrtdata/carla360')

    if args.func == 'create_waymo_infos':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            kitti_path=KITTI_PATH,
            data_path=ROOT_DIR / 'carla360_waymo',
            save_path=ROOT_DIR / 'carla360_waymo',
            segment_info_tag='segment_info',
            processed_data_tag=args.processed_data_tag,
            update_info_only=args.update_info_only,
            carla_data=args.carla_data
        )
    elif args.func == 'create_waymo_gt_database':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        create_waymo_gt_database(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'carla360_waymo',
            save_path=ROOT_DIR / 'carla360_waymo',
            processed_data_tag=args.processed_data_tag,
            use_parallel=args.use_parallel, 
            crop_gt_with_tail=not args.wo_crop_gt_with_tail,
            carla_data=args.carla_data
        )
    elif args.func == 'collect_frame_to_sequence':
        collect_frame_to_sequence(
            KITTI_PATH, 
            ROOT_DIR / 'carla360_waymo',
            segment_info_tag='segment_info',
            carla_data=args.carla_data,
            sequence_length=args.sequence_length)
    else:
        raise NotImplementedError

