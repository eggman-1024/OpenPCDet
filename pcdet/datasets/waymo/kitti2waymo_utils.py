import os
import pickle
import numpy as np
import math
from ...utils import common_utils, calibration_kitti, box_utils


def euler2mat(roll, pitch, yaw, degrees=True):
        '''
        欧拉角转旋转矩阵
        :param roll: 滚转角
        :param pitch: 俯仰角
        :param yaw: 偏航角
        :param degrees: 是否为角度制
        :return:
        '''
        # 角度制转弧度制
        if degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

        cos_r, sin_r = math.cos(roll), math.sin(roll)
        R_x = np.array([[1, 0, 0],
                        [0, cos_r, -sin_r],
                        [0, sin_r, cos_r]])

        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        R_y = np.array([[cos_p, 0, sin_p],
                        [0, 1, 0],
                        [-sin_p, 0, cos_p]])

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R_z = np.array([[cos_y, -sin_y, 0],
                        [sin_y, cos_y, 0],
                        [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def trans_class_name_to_waymo(carla_class_name):
    if carla_class_name == 'Car':
        return 'Vehicle'
    elif carla_class_name == 'DontCare':
        return 'unknown'
    else:
        return carla_class_name
    

def get_pose(status_path):
    assert os.path.exists(status_path)

    with open(status_path, 'r') as f:
        poses = f.readlines()[1]
        # 内容是 "lidar: x y z roll pitch yaw\n"
        # 去除前面的lidar:和后面的\n
        poses = poses.replace('lidar: ', '').replace('\n', '')
        poses = poses.split(" ")
        poses = np.array(poses, dtype=np.float32)

    roll, pitch, yaw = poses[3:].astype(np.float32)
    R = euler2mat(roll, pitch, yaw)
    T = poses[:3].astype(np.float32)

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T

    return transform


def get_timestamp(status_path):
    # 内容是 "timestamp: 15.27984008193016\n"
    with open(status_path, 'r') as f:
        timestamp = f.readlines()[0].strip()
        timestamp = float(timestamp.split(' ')[1])
    return timestamp

def get_calib(calib_file):
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)


def save_lidar_points(points, cur_save_path):
    '''
    保存点云数据，存为.npy文件
    :param points: 点云数据
    :param cur_save_path: 保存路径
    :return: 点云数量
    '''
    np.save(cur_save_path, points)
    return points.shape[0]

def process_single_sequence(sequence_info, kitti_path, save_path, sampled_interval=1,
                            carla_data=True, has_label=True, update_info_only=False):
    sequence_infos = []
    # segment_000_000000 -> segment_000
    sequence_name = sequence_info[0]['frame_id'].split('_')[:-1]
    sequence_name = '_'.join(sequence_name)

    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    for cnt, data in enumerate(sequence_info):
        frame_id = data['frame_id']
        raw_frame_id = data['raw_frame_id']

        calib_path = os.path.join(kitti_path, 'training', 'calib', raw_frame_id + '.txt')
        image_path = os.path.join(kitti_path, 'training', 'image_2', raw_frame_id + '.png')
        label_path = os.path.join(kitti_path, 'training', 'label_2', raw_frame_id + '.txt')
        if carla_data:
            status_path = os.path.join(kitti_path, 'training', 'status', raw_frame_id + '.txt')
        velodyne_path = os.path.join(kitti_path, 'training', 'velodyne', raw_frame_id + '.bin')

        pc = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 12)


        info = {}
        pc_info = {'num_features': 12, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': sequence_name,                  # 不知有何作用
            'timestamp_micros': get_timestamp(status_path)  # timestamp
        }
        # 不添加image_info

        pose = get_pose(status_path)    # lidar to world transform (4x4)
        info['pose'] = pose
        if has_label:
            annotations = generate_labels(info['frame_id'], calib_path, label_path, pc)
            info['annos'] = annotations
        
        num_points_of_each_lidar = save_lidar_points(pc, cur_save_dir / ('%04d.npy' % cnt))
        info['num_points_of_each_lidar'] = [num_points_of_each_lidar]

        sequence_infos.append(info)
    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)
    return sequence_infos


def generate_labels(frame_id, calib_path, label_path, points):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []

    calib = calibration_kitti.Calibration(calib_path)

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for cnt, line in enumerate(lines):
            line = line.strip().split(' ')
            obj_name.append(trans_class_name_to_waymo(line[0]))
            difficulty.append(0)                                # 后续会根据gt框点云数量更改为1或2, 现在需要赋值为0
            tracking_difficulty.append(0)                       # 用不到tracking 所以赋值为0

            # line[8:11]需要倒过来遍历
            l, w, h = [float(x) for x in line[8:11][::-1]]
            dimensions.append([float(x) for x in line[8:11][::-1]]) # kitti格式是hwl, waymo格式是lwh

            x, y, z = [float(x) for x in line[11:14]]
            y -= h / 2  # kitti的z是物体底部, waymo的z是物体中心(相机坐标系下的y是向下的)
            locations.append([x, y, z])
            heading_angles.append(float(line[14]))

            # obj_ids在waymo中是类似 -1h4WDWtuP9P8-ULyuPo-Q 的字符串, 此处暂时用frame_id和cnt拼接
            obj_ids.append('{}_{:03d}'.format(frame_id, cnt))

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(calib.rect_to_lidar(np.array(locations)))    # camera to lidar
    annotations['heading_angles'] = np.array(-(np.pi / 2 + np.array(heading_angles))) # camera to lidar

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    gt_boxes_lidar = np.concatenate([annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis]], axis=1)
    

    num_objects = len(obj_name)
    num_points_in_gt = -np.ones(num_objects, dtype=np.int32)
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    for k in range(num_objects):
        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
        num_points_in_gt[k] = flag.sum()
    annotations['num_points_in_gt'] = num_points_in_gt
    annotations['gt_boxes_lidar'] = np.concatenate([gt_boxes_lidar, np.zeros((num_objects, 2))], axis=-1)
    return annotations


def collect_frame_to_sequence(kitti_path, output_path, segment_info_tag, carla_data=True, sequence_length=80):
    # kitti_path必须要绝对路径
    print(f"=========Collecting frame to sequence from {kitti_path} to {output_path} and generate ImageSets=========")
    os.makedirs(os.path.join(output_path, segment_info_tag), exist_ok=True)
    
    training_path = os.path.join(kitti_path, 'training')

    calib_path = os.path.join(training_path, 'calib')
    image_path = os.path.join(training_path, 'image_2')
    label_path = os.path.join(training_path, 'label_2')
    if carla_data:
        status_path = os.path.join(training_path, 'status') # timestamp, lidar pose, ego pose
    velodyne_path = os.path.join(training_path, 'velodyne') # .bin

    assert len(os.listdir(image_path)) == len(os.listdir(label_path)) == len(os.listdir(velodyne_path)) == len(os.listdir(calib_path))

    # 每80帧为一个sequence，创建一个dict来存储这个sequence的信息，最后输出为pickle文件
    for seq_id in range(len(os.listdir(velodyne_path)) // sequence_length):
        sequence_info = []
        # 将seq_id扩展为3位数的str
        seq_name = "segment_{:03d}".format(seq_id)
        # 读取每个seq内的数据
        for i in range(sequence_length):
            frame_info = {}
            frame_name = "{:06d}".format(i)
            frame_id = f"{seq_name}_{frame_name}" # segment_000_000000

            raw_frame_name = "{:06d}".format(i + seq_id * sequence_length)

            frame_info['frame_id'] = frame_id
            frame_info['raw_frame_id'] = raw_frame_name
            # frame_info['calib_path'] = os.path.join(calib_path, frame_name + '.txt')
            # frame_info['image_path'] = os.path.join(image_path, frame_name + '.png')
            # frame_info['label_path'] = os.path.join(label_path, frame_name + '.txt')
            # if carla_data:
            #     frame_info['status_path'] = os.path.join(status_path, frame_name + '.txt')
            # frame_info['velodyne_path'] = os.path.join(velodyne_path, frame_name + '.bin')

            sequence_info.append(frame_info)
        
        # 保存为pickle文件
        with open(os.path.join(output_path, segment_info_tag, f'{seq_name}.pkl'), 'wb') as f:
            pickle.dump(sequence_info, f)
    
    

    # 生成ImageSets文件夹
    os.makedirs(os.path.join(output_path, 'ImageSets'), exist_ok=True)
    kitti_imagesets_path = os.path.join(kitti_path, 'ImageSets')
    with open(os.path.join(kitti_imagesets_path, 'train.txt'), 'r') as f:
        train_segment_length = len(f.readlines()) // sequence_length
        with open(os.path.join(output_path, 'ImageSets', 'train.txt'), 'w') as f_out:
            f_out.write('\n'.join(['segment_{:03d}'.format(i) for i in range(train_segment_length)]))

    with open(os.path.join(kitti_imagesets_path, 'val.txt'), 'r') as f:
        val_segment_length = len(f.readlines()) // sequence_length
        with open(os.path.join(output_path, 'ImageSets', 'val.txt'), 'w') as f_out:
            f_out.write('\n'.join(['segment_{:03d}'.format(i) for i in range(train_segment_length, train_segment_length + val_segment_length)]))
    
    