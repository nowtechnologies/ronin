import json
import random
from os import path as osp

import h5py
import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences


class GlobSpeedSequence(CompiledSequence):
    """
    Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    # from data_utils.load_cached_sequences as seq = seq_type(osp.join(root_dir, data_list[i]), **kwargs)
    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1) # 200
        if data_path is not None:
            self.load(data_path)

    # from __init__
    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        with open(osp.join(data_path, 'info.json')) as f:
            self.info = json.load(f)

        # {'tango_reference_time': 3697949488366.0, 'date': '01/21/19', 'imu_init_gyro_bias': [0.005569000000000001, 0.009201, -0.00018299999999999914], 'imu_acce_scale': [0.9987918889533104, 0.997062129866083, 0.9932574091553678], 'grv_ori_error': 8.353779492444412, 'align_tango_to_body': [-0.4841758535176575, -0.4938374978693588, -0.5424326634072199, 0.47693298715598254], 'start_frame': 5896, 'imu_end_gyro_bias': [0.005432, 0.008774, -4.6e-05], 'type': 'annotated', 'imu_reference_time': 3610942440968.0, 'start_calibration': [0.0, 0.99995545, 0.009439, 0.0], 'ekf_ori_error': 3.6239102945197676, 'imu_acce_bias': [-0.15661902624553775, -0.026333329541761968, 0.05681364453654479], 'gyro_integration_error': 8.481689436777705, 'device': 'asus4', 'length': 323.7550000070669, 'imu_time_offset': -0.07619643889847794, 'end_calibration': [0.0, 0.99998599, 0.0052938, 0.0]}

        self.info['path'] = osp.split(data_path)[-1]

        self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source( # ori is hdf5.synced.game_rv i. e. Android Sensor.TYPE_GAME_ROTATION_VECTOR
            data_path, self.max_ori_error, self.grv_only)
        #('game_rv', array([[ 0.02301384, -0.734161  ,  0.00956859,  0.67851714], [ 0.02296023, -0.73417201,  0.00956628,  0.67850771], ..., [ 0.05427992,  0.70881762,  0.07637797, -0.6991363 ]]]), 8.353779492444412)

        with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
            gyro_uncalib = f['synced/gyro_uncalib']                                                                # <HDF5 dataset "gyro_uncalib": shape (64752, 3), type "<f8">
            acce_uncalib = f['synced/acce']                                                                        # <HDF5 dataset "acce": shape (64752, 3), type "<f8">
            gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])                                        # array([[-0.02230625,  0.01376489,  0.01876768], [-0.01673078,  0.00785385,  0.01808999], ..., [ 0.00657855, -0.02807542,  0.00804713]])
            acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))   # array([[-9.76895341, -0.19332236, -0.85234999], [-9.76140265, -0.2099069 , -0.81018915], ..., [-9.82066284, -0.32593967, -0.28265888]])
            ts = np.copy(f['synced/time'])                                                                         # array([3641.39639807, 3641.40139807, 3641.40639807, ..., 3965.14139807, 3965.14639807, 3965.15139807])
            tango_pos = np.copy(f['pose/tango_pos'])                                                               # array([[ 0.00747055,  0.0794231 ,  0.04721916], [ 0.00743938,  0.07954534,  0.04721213], ..., [ 0.04869788, -0.01891041, -0.03532039]])
            init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])                                        # quaternion(0.500218919180744, 0.498520940104168, 0.458115146317795, -0.539803994906776)

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)                                   # array([quaternion(0.0230138445473811, -0.734161004581412, 0.00956858773770847, 0.678517142961637), quaternion(0.0229602307881793, -0.734172007659053, 0.00956628356173319, 0.678507709011024), ..., quaternion(0.0542799190079345, 0.708817623072373, 0.0763779734707989, -0.6991362963311)], dtype=quaternion)
        # hdf5.synced.game_rv i. e. Android Sensor.TYPE_GAME_ROTATION_VECTOR

        rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])  # quaternion(0, 0.99995545, 0.009439, 0)
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()           # quaternion(-0.695289552060529, 0.00118374652078425, 0.00248606386725569, 0.718723783950829)
        ori_q = init_rotor * ori_q                                                 # array([quaternion(-0.5028224217168, 0.50529138394065, -0.535057892743795, -0.453388785030156), quaternion(-0.502778345472314, 0.505300603413145, -0.535064320967734, -0.453420734559951), ..., quaternion(0.463716682908265, -0.54940199756135, 0.457301820727505, 0.523442677366401)], dtype=quaternion)
        # ori_q = f['pose/tango_ori'][0] * self.info['start_calibration'] * conj(f['synced/game_rv'][0]) * f['synced/game_rv']

        dt = (ts[self.w:] - ts[:-self.w])[:, None]                                 # array([[1.], [1.], [1.], ..., [1.],[1.], [1.]])

        glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt                   # array([[-0.00533056,  0.01667982, -0.00509732], [-0.00531125,  0.016594  , -0.00511179], ..., [-0.0023489 ,  0.00633583, -0.00057166]])

        # these two below are position vector arrays
        gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))  # array([quaternion(0, -0.0223062507369463, 0.0137648946088728, 0.0187676819234896), quaternion(0, -0.0167307794589504, 0.00785385399152264, 0.018089989505323), ..., quaternion(0, 0.00657855020053501, -0.0280754170707831, 0.0080471337151681)], dtype=quaternion)
        acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))  # array([quaternion(0, -9.76895340810378, -0.193322357928738, -0.852349993053583), quaternion(0, -9.76140265037272, -0.209906902110648, -0.810189151712629), ..., quaternion(0, -9.8206628384554, -0.325939671417927, -0.282658875290474)], dtype=quaternion)

        # each element vector rotated by the corresponding ori_q rotation quaternion
        # At test time, we use the coordinate frame defined by system device orientations from Android or iOS, whose Z axis is aligned with gravity.
        # The whole is transformed into the global (Tango) coordinate frame.
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]  # array([[-0.01258328,  0.02161022,  0.02034508], [-0.00665554,  0.02000185,  0.0149826 ], ..., [ 0.02674353,  0.01209917, -0.0058859 ]])
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]  # array([[-3.46650073e-02, -3.36474233e-02,  9.80783499e+00], [-1.38849800e-02,  6.54256497e-03,  9.79719413e+00], ..., [ 3.31727211e-02, -6.26925428e-02,  9.82980926e+00]])

        start_frame = self.info.get('start_frame', 0)                                # 5896
        self.ts = ts[start_frame:]                                                   # array([3670.87639807, 3670.88139807, 3670.88639807, ..., 3965.14139807, 3965.14639807, 3965.15139807])
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:] # array([[-5.94662071e-03, -9.38751552e-03, -5.15188486e-03, -3.08588928e-02,  6.39869105e-02,  9.86019268e+00], [-5.27530580e-03, -7.75847573e-03, -1.59778536e-02, -3.54599110e-02,  5.16253587e-02,  9.82394159e+00], ..., [ 2.67435308e-02,  1.20991655e-02, -5.88589595e-03, 3.31727211e-02, -6.26925428e-02,  9.82980926e+00]])
        self.targets = glob_v[start_frame:, :2]                                      # array([[-0.02427537,  0.02117807], [-0.02406481,  0.02145767], ..., [-0.0023489 ,  0.00633583]]) targets are averaged for the window w
        self.orientations = quaternion.as_float_array(ori_q)[start_frame:]           # array([[-0.51946022,  0.24279321, -0.60182678,  0.55589147], [-0.51947897,  0.24272502, -0.6018242 ,  0.55590699], ..., [ 0.46371668, -0.549402  ,  0.45730182,  0.52344268]])
        self.gt_pos = tango_pos[start_frame:]                                        # array([[ 0.17387274, -0.14344794, -0.0743621 ], [ 0.17362087, -0.14350179, -0.07425673], ..., [ 0.04869788, -0.01891041, -0.03532039]])

    # from data_utils.load_cached_sequences as feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
    def get_feature(self):
        return self.features   # [ 3 from glob_gyro; 3 from glob_acce ]*

    # from data_utils.load_cached_sequences as feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
    def get_target(self):
        return self.targets

    # from data_utils.load_cached_sequences as feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
    def get_aux(self):
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)  # array([[ 3.67087640e+03, -5.19460219e-01,  2.42793212e-01, ..., 1.73872741e-01, -1.43447937e-01, -7.43620958e-02], [ 3.67088140e+03, -5.19478970e-01,  2.42725017e-01, ..., 1.73620867e-01, -1.43501792e-01, -7.42567343e-02], ..., [ 3.96515140e+03,  4.63716683e-01, -5.49401998e-01, ..., 4.86978752e-02, -1.89104115e-02, -3.53203909e-02]])
        # .shape(58856, 8)
        # vmi[0:2,:] array([[ 3.67087640e+03, -5.19460219e-01,  2.42793212e-01, -6.01826777e-01,  5.55891467e-01,  1.73872741e-01, -1.43447937e-01, -7.43620958e-02],
        #                   [ 3.67088140e+03, -5.19478970e-01,  2.42725017e-01, -6.01824198e-01,  5.55906995e-01,  1.73620867e-01, -1.43501792e-01, -7.42567343e-02]])



    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=1, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class StridedSequenceDataset(Dataset):
    # from ronin_resnet.get_dataset as dataset = StridedSequenceDataset
    #                  (seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size, random_shift=random_shift, transform=transforms, shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)
    def __init__(self,  seq_type, root_dir, data_list, cache_path=None, step_size=10,    window_size=200, random_shift=0,            transform=None,       **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size) # 200

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(                            # self.features contains all the test-ready input data as 3D vectors [ 3 gyro, 3 acce ]*
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)] # [seq0, 0] [seq0, 10] ... [seq0, target0.len] [seq1, 0] [seq1, 10] ... 

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]           # 200 elements from [ 3 gyro, 3 acce ]* starting at frame_id which increments by 10 on each query
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id    # .T is transpose

    def __len__(self):
        return len(self.index_map)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=400,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, 5:8])

            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)
