import sys
import cv2
import time
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os

#from smplx import SMPL as _SMPL
from smplx import SMPL
from smplx.lbs import vertices2joints
from smplx.body_models import ModelOutput
import numpy as np
import torch

SMOOTHING_WINDOW = None
SMPL_MODEL_DIR = 'vibert/data/vibe_data'

def distance(x1, y1, x2, y2):
    l = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    return l

def triangle_square(x1, y1, x2, y2, x3, y3):
    square = abs( x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2) ) / 2.0
    return square

def indices(a_list: list, int_value: int) -> list:
    return [index for index, element in enumerate(a_list) if element == int_value]

def value(val: int, a_l: type) -> str:
    return int(','.join(map(str,indices(a_l, val))))

def main():
    src_path = sys.argv[1]

    cam6_in = joblib.load(os.path.join(src_path, 'cam6.pkl'))
    cam5_in = joblib.load(os.path.join(src_path, 'cam5.pkl'))
    cam4_in = joblib.load(os.path.join(src_path, 'cam4.pkl'))
    cam1_in = joblib.load(os.path.join(src_path, 'cam1.pkl'))
    cam2_in = joblib.load(os.path.join(src_path, 'cam2.pkl'))
    cam3_in = joblib.load(os.path.join(src_path, 'cam3.pkl'))

    print("SMPL dicts: ")
    print(cam6_in.keys())
    print(cam5_in.keys())
    print(cam4_in.keys())
    print(cam1_in.keys())
    print(cam2_in.keys())
    print(cam3_in.keys())
    print("\n")

    cam6_len = len(cam6_in[0]['joints2d'])
    cam5_len = len(cam5_in[0]['joints2d'])
    cam4_len = len(cam4_in[0]['joints2d'])
    cam1_len = len(cam1_in[0]['joints2d'])
    cam2_len = len(cam2_in[0]['joints2d'])
    cam3_len = len(cam3_in[0]['joints2d'])

    print("Length of sequence with [0] idx: ")
    print("cam6_len: " + str(cam6_len))
    print("cam5_len: " + str(cam5_len))
    print("cam4_len: " + str(cam4_len))
    print("cam1_len: " + str(cam1_len))
    print("cam2_len: " + str(cam2_len))
    print("cam3_len: " + str(cam3_len))

    seq_len = min(cam6_len, cam5_len, cam4_len, cam1_len, cam2_len, cam3_len)
    print("Minimal sequence length for truncation: " + str(seq_len))

    # prepare joints array for prediction from each camera position
    J2D = []
    J2D.append(cam6_in[0]['joints2d'])
    J2D.append(cam5_in[0]['joints2d'])
    J2D.append(cam4_in[0]['joints2d'])
    J2D.append(cam1_in[0]['joints2d'])
    J2D.append(cam2_in[0]['joints2d'])
    J2D.append(cam3_in[0]['joints2d'])

    # Find more frontal camera index from area of shoulder-hip zone projection sizes
    # normalised by Y projection size of skeleton
    idx_frontal = []
    idx3_frontal = []
    idx3_medium = []
    idx_cam = []

    for i in range(seq_len):
        max_square = 0
        area_array = []

        # find index of maximum projected shoulder distance
        for k in range(len(J2D)):
            I = J2D[k][i]
            # left shouder
            x1 = I[2][0]
            y1 = I[2][1]
            # right shoulder
            x2 = I[5][0]
            y2 = I[5][1]
            # left hip
            x3 = I[9][0]
            y3 = I[9][1]
            # right hip
            x4 = I[12][0]
            y4 = I[12][1]

            y_pnts = [y1, y2, y3, y4]

            y_min = min(y_pnts)
            y_max = max(y_pnts)
            norm_l = y_max - y_min
            # print(y_min, y_max, norm_l)

            tri1_square = triangle_square(x1, y1, x2, y2, x3, y3)
            tri2_square = triangle_square(x4, y4, x2, y2, x3, y3)
            orient = (x2 - x1) / abs(x1 - x2)
            sum_square = (tri1_square + tri2_square) * orient
            # norm_square = sum_square
            norm_square = sum_square / norm_l

            # single maximum area search
            if norm_square > max_square:
                max_square = norm_square
                max_idx = k

            area_array.append(abs(norm_square))

        sorted_arr = sorted(set(area_array))
        #print(sorted_arr)

        idx_frontal.append(max_idx)
        max3_idxs = [value(sorted_arr[-1], area_array), value(sorted_arr[-2], area_array),
                     value(sorted_arr[-3], area_array)]
        #print(max3_idxs)
        med3_idxs = [value(sorted_arr[-2], area_array), value(sorted_arr[-3], area_array),
                     value(sorted_arr[-4], area_array)]
        #print(med3_idxs)

        idx3_frontal.append(max3_idxs)
        idx3_medium.append(med3_idxs)
        #idx_cam.append(cam_idx2num[max_idx])

        #print(max_idx)
        #print("\n\n")

    #print("Frontal indexes per 1 max: ")
    #print(idx_frontal)
    #print("Camera numbers per 1 max: ")
    #print(idx_cam)

    # print("Frontal indexes per 3 max: ")
    # print(idx3_frontal)
    #
    # print("Medium indexes per 3 max: ")
    # print(idx3_medium)

    cams_in=[]
    cams_in.append(cam6_in[0])
    cams_in.append(cam5_in[0])
    cams_in.append(cam4_in[0])
    cams_in.append(cam1_in[0])
    cams_in.append(cam2_in[0])
    cams_in.append(cam3_in[0])

    N = len(cams_in)
    for i in range(seq_len - 1):
        # blending shapes part of SMPL vector
        sum_shape = 0
        for k in range(N):
            sum_shape += cams_in[k]["betas"][i]
        med_shape = sum_shape / N

        # blending pose part of SMPL vector
        idx0 = idx3_frontal[i][0]
        idx1 = idx3_frontal[i][1]
        idx2 = idx3_frontal[i][2]

        midx0 = idx3_medium[i][0]
        midx1 = idx3_medium[i][1]
        midx2 = idx3_medium[i][2]

        # 3 idx max averaging
        med_pose = (cams_in[idx0]["pose"][i][3:] + cams_in[idx1]["pose"][i][3:] + cams_in[idx2]["pose"][i][3:]) / 3.0
        med_rotmat = (cams_in[idx0]["rotmat"][i][1:] + cams_in[idx1]["rotmat"][i][1:] + cams_in[idx2]["rotmat"][i][1:]) / 3.0

        # 2 idx averaging (except 1 frontal)
        # med_pose = (cams_in[idx1]["pose"][i][3:] + cams_in[idx2]["pose"][i][3:])/2.0
        # med_rotmat = (cams_in[idx1]["rotmat"][i][1:] + cams_in[idx2]["rotmat"][i][1:])/2.0

        # 3 idx medium averaging
        # med_pose = (cams_in[midx0]["pose"][i][3:] + cams_in[midx1]["pose"][i][3:] + cams_in[midx2]["pose"][i][3:])/3.0
        # med_rotmat = (cams_in[midx0]["rotmat"][i][1:] + cams_in[midx1]["rotmat"][i][1:] + cams_in[midx2]["rotmat"][i][1:])/3.0

        # only 1 (frontal or not) idx using
        # med_pose = cams_in[idx1]["pose"][i][3:]
        # med_rotmat = cams_in[idx1]["rotmat"][i][1:]

        # renew the predicts with averaged values
        for k in range(N):
            cams_in[k]["betas"][i] = med_shape
            cams_in[k]["pose"][i][3:] = med_pose
            cams_in[k]["rotmat"][i][1:] = med_rotmat


    smpl = SMPL(SMPL_MODEL_DIR, batch_size=seq_len, create_transl=False, gender='male')
    for k in range(N):
        print(cams_in[k]["rotmat"].shape)
        torch_betas = torch.from_numpy(cams_in[k]["betas"])
        pred_rotmat = torch.from_numpy(cams_in[k]["rotmat"])
        print(pred_rotmat.shape)
        pred_output = smpl(betas=torch_betas,
                           body_pose=pred_rotmat[:, 1:],
                           global_orient=pred_rotmat[:, 0].unsqueeze(1),
                           pose2rot=False)
        #cams_in[k]["verts"] = pred_output.vertices.cpu().numpy()

    def smooth_1d(x, SMOOTHING_WINDOW=30):
      # from [-pi, pi] to [0, 2pi]
      x += np.pi

      # find discontinuities
      c = np.zeros_like(x, np.float32)
      c[np.where(np.diff(x) < -np.pi)[0]] = 2 * np.pi
      c[np.where(np.diff(x) > np.pi)[0]] = -2 * np.pi

      # from mod 2pi to continuous
      x += np.cumsum(c)

      # smoothing
      x = np.convolve(x, [1.0 / SMOOTHING_WINDOW] * SMOOTHING_WINDOW, "same")

      # from continuous to mod 2pi
      x = x % (2 * np.pi)

      return x - np.pi

    for k in range(N):
        if SMOOTHING_WINDOW is not None and cams_in[k]["pose"].shape[0] > SMOOTHING_WINDOW:
            cams_in[k]["pose"] = np.apply_along_axis(smooth_1d, 0, cams_in[k]["pose"]).astype(np.float32)

            # cams_in[k]["pose"] = np.apply_along_axis(lambda x: np.convolve(x, [1.0 / SMOOTHING_WINDOW] * SMOOTHING_WINDOW, "same"), 0, cams_in[k]["pose"]).astype(np.float32)

        print(cams_in[k]["pose"].shape)
        torch_betas = torch.from_numpy(cams_in[k]["betas"])
        pred_pose = torch.from_numpy(cams_in[k]["pose"])

        print(pred_pose.shape)
        pred_output = smpl(betas=torch_betas,
                           body_pose=pred_pose[:, 3:],
                           global_orient=pred_pose[:, :3],
                           return_full_pose=True)
        cams_in[k]["verts"] = pred_output.vertices.cpu().numpy()

    # Truncate other dictionary arrays for len = seq_len
    for k in range(N):
        cams_in[k]["pred_cam"] = cams_in[k]["pred_cam"][:seq_len-1]
        cams_in[k]["orig_cam"] = cams_in[k]["orig_cam"][:seq_len-1]
        cams_in[k]["pose"] = cams_in[k]["pose"][:seq_len-1]
        cams_in[k]["betas"] = cams_in[k]["betas"][:seq_len-1]
        cams_in[k]["verts"] = cams_in[k]["verts"][:seq_len-1]
        cams_in[k]["joints3d"] = cams_in[k]["joints3d"][:seq_len-1]
        cams_in[k]["joints2d"] = cams_in[k]["joints2d"][:seq_len-1]
        cams_in[k]["bboxes"] = cams_in[k]["bboxes"][:seq_len-1]
        cams_in[k]["frame_ids"] = cams_in[k]["frame_ids"][:seq_len-1]
        # added from internal of prediction code:
        cams_in[k]["n_joints2d"] = cams_in[k]["n_joints2d"][:seq_len-1]
        cams_in[k]["rotmat"] = cams_in[k]["rotmat"][:seq_len-1]

    cam6_out = {0: cams_in[0]}
    cam5_out = {0: cams_in[1]}
    cam4_out = {0: cams_in[2]}
    cam1_out = {0: cams_in[3]}
    cam2_out = {0: cams_in[4]}
    cam3_out = {0: cams_in[5]}

    # Storing results
    import pickle
    with open(os.path.join(src_path, 'cam6_fin.pkl'), 'wb') as f:
        pickle.dump(cam6_out, f)
    with open(os.path.join(src_path, 'cam5_fin.pkl'), 'wb') as f:
        pickle.dump(cam5_out, f)
    with open(os.path.join(src_path, 'cam4_fin.pkl'), 'wb') as f:
        pickle.dump(cam4_out, f)
    with open(os.path.join(src_path, 'cam1_fin.pkl'), 'wb') as f:
        pickle.dump(cam1_out, f)
    with open(os.path.join(src_path, 'cam2_fin.pkl'), 'wb') as f:
        pickle.dump(cam2_out, f)
    with open(os.path.join(src_path, 'cam3_fin.pkl'), 'wb') as f:
        pickle.dump(cam3_out, f)

if __name__ == '__main__':
  main()
