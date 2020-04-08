import os
import numpy as np
import argparse
import torch
from lib.models.spin import SMPL
from holo.data_struct import DataStruct
from lib.utils.geometry import rotation_matrix_to_angle_axis


def find_frontal_camera(J2D, seq_len):
    '''
    Find more frontal camera index from area of shoulder-hip zone
    projection sizes normalised by Y projection size of skeleton
    '''
    def triangle_square(x1, y1, x2, y2, x3, y3):
        square = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        return square

    idx3_frontal = []

    for i in range(seq_len):
        area_array = []

        # find index of maximum projected shoulder distance
        for key,val in J2D.items():
            # left shouder
            x1 = val[i][2][0]
            y1 = val[i][2][1]
            # right shoulder
            x2 = val[i][5][0]
            y2 = val[i][5][1]
            # left hip
            x3 = val[i][9][0]
            y3 = val[i][9][1]
            # right hip
            x4 = val[i][12][0]
            y4 = val[i][12][1]

            y_pnts = [y1, y2, y3, y4]
            y_min = min(y_pnts)
            y_max = max(y_pnts)
            norm_l = y_max - y_min

            tri1_square = triangle_square(x1, y1, x2, y2, x3, y3)
            tri2_square = triangle_square(x4, y4, x2, y2, x3, y3)
            orient = (x2 - x1) / abs(x1 - x2)
            sum_square = (tri1_square + tri2_square) * orient
            norm_square = sum_square / norm_l
            area_array.append((abs(norm_square),key))

        sorted_arr = sorted(area_array, key=lambda x: x[0])
        idx3_frontal.append([sorted_arr[-1][1], sorted_arr[-2][1], sorted_arr[-3][1]])

    return idx3_frontal


def align_smpl(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get data struct:
    output_dir = os.path.join(args.root_dir, args.result_dir)
    pkl_path = os.path.join(args.root_dir, args.smpl_dir)
    data_smpl = DataStruct().parse(pkl_path, levels='subject/light/garment/scene/cam', ext='npz')

    # Run alignment for all persons:
    for node, path in data_smpl.nodes('scene'):
        print ('Processing dir', path)
        smpl_in = {}
        for cam in data_smpl.items(node):
            cam_name = cam.dir.split('/')[-1]
            with np.load(cam.abs_path, encoding='latin1', allow_pickle=True) as data:
                smpl_in[cam_name] = dict(data)
        assert len(smpl_in) == 6, (len(smpl_in))

        if args.trim_frames:
            n = min([smpl['frame_paths'].shape[0] for smpl in smpl_in.values()])
            for k1 in smpl_in.keys():
                for k2 in smpl_in[k1].keys():
                    smpl_in[k1][k2] = smpl_in[k1][k2][:n]

        # Find the most frontal camera indices:
        J2D = {k : v['n_joints2d'] for k,v in smpl_in.items()}
        seq_len = len(list(J2D.values())[0])
        idx3_frontal = find_frontal_camera(J2D, seq_len)

        # Average the SMPL vectors obtained from different cameras:
        for frame_i in range(seq_len):
            # Blend SMPL shapes:
            sum_shape = 0
            for smpl in smpl_in.values():
                sum_shape += smpl['betas'][frame_i]
            med_shape = sum_shape / len(smpl_in)

            # Blend SMPL poses:
            idx0 = idx3_frontal[frame_i][0]
            idx1 = idx3_frontal[frame_i][1]
            idx2 = idx3_frontal[frame_i][2]
            med_pose = (smpl_in[idx0]['pose'][frame_i] +
                        smpl_in[idx1]['pose'][frame_i] +
                        smpl_in[idx2]['pose'][frame_i]) / 3.0
            med_rotmat = (smpl_in[idx0]['rotmat'][frame_i] +
                          smpl_in[idx1]['rotmat'][frame_i] +
                          smpl_in[idx2]['rotmat'][frame_i]) / 3.0

            # Renew the predicts with averaged values:
            for k in smpl_in.keys():
                smpl_in[k]['betas'][frame_i][:] = med_shape
                # smpl_in[k]['pose'][frame_i][3:] = med_pose[3:]
                smpl_in[k]['rotmat'][frame_i][1:] = med_rotmat[1:]

                # Fix a pose by rotmat:
                if args.pose_by_rotmat:
                    rotmat = torch.from_numpy(smpl_in[k]['rotmat'][frame_i])
                    pose = rotation_matrix_to_angle_axis(rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
                    smpl_in[k]['pose'][frame_i] = pose.cpu().numpy()

        # Regenerate SMPL vertices:
        if args.include_vertices:
            smpl_verts = {}
            smpl_model = SMPL(args.vibe_model_dir, batch_size=seq_len, create_transl=False, gender=args.gender)
            smpl_model.to(device)
            for k in smpl_in.keys():
                betas = torch.from_numpy(smpl_in[k]['betas']).to(device)
                rotmat = torch.from_numpy(smpl_in[k]['rotmat']).to(device)
                pred_output = smpl_model(betas=betas,
                                         body_pose=rotmat[:, 1:],
                                         global_orient=rotmat[:, 0].unsqueeze(1),
                                         pose2rot=False)
                smpl_in[k]['verts'] = pred_output.vertices.detach().cpu().numpy()

        # Save the averaged SMPLs:
        for cam in data_smpl.items(node):
            cam_name = cam.dir.split('/')[-1]
            result_dir = os.path.join(output_dir, path, cam_name)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_path = os.path.join(result_dir, 'smpl.npz')

            if args.output_format == 'LWGAN':
                cams = smpl_in[cam_name]['avatar_cam']
                if cams.shape[1] > 3:
                    cams = np.stack((cams[:, 0], cams[:, 2], cams[:, 3]), axis=1)
                assert cams.shape[1] == 3
                np.savez(result_path,
                         cams=cams,
                         pose=smpl_in[cam_name]['pose'],
                         shape=smpl_in[cam_name]['betas'])
            else:
                if args.include_vertices:
                    np.savez(result_path,
                             avatar_cam=smpl_in[cam_name]['avatar_cam'],
                             avatar_bboxes=smpl_in[cam_name]['bboxes'],
                             pred_cam=smpl_in[cam_name]['pred_cam'],
                             orig_cam=smpl_in[cam_name]['orig_cam'],
                             pose=smpl_in[cam_name]['pose'],
                             betas=smpl_in[cam_name]['betas'],
                             joints3d=smpl_in[cam_name]['joints3d'],
                             n_joints2d=smpl_in[cam_name]['n_joints2d'],
                             rotmat=smpl_in[cam_name]['rotmat'],
                             bboxes=smpl_in[cam_name]['bboxes'],
                             frame_paths=smpl_in[cam_name]['frame_paths'],
                             verts=smpl_in[cam_name]['verts'])
                else:
                    np.savez(result_path,
                             avatar_cam=smpl_in[cam_name]['avatar_cam'],
                             avatar_bboxes=smpl_in[cam_name]['bboxes'],
                             pred_cam=smpl_in[cam_name]['pred_cam'],
                             orig_cam=smpl_in[cam_name]['orig_cam'],
                             pose=smpl_in[cam_name]['pose'],
                             betas=smpl_in[cam_name]['betas'],
                             joints3d=smpl_in[cam_name]['joints3d'],
                             n_joints2d=smpl_in[cam_name]['n_joints2d'],
                             rotmat=smpl_in[cam_name]['rotmat'],
                             bboxes=smpl_in[cam_name]['bboxes'],
                             frame_paths=smpl_in[cam_name]['frame_paths'])

    print('All done!')


def main():
    # Set params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/darkalert/KazendiJob/Data/HoloVideo/Data',
                        help='root dir path')
    parser.add_argument('--smpl_dir', type=str, default='smpl_maskrcnn_t2',
                        help='dir containing input smpl pkls')
    parser.add_argument('--result_dir', type=str, default='smpl_maskrcnn_aligned',
                        help='dir that will contain the result of smpl alignment')
    parser.add_argument('--vibe_model_dir', type=str, default='vibert/data/vibe_data',
                        help='smpl model dir for VIBE')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='gender of smpl for rendering')
    parser.add_argument('--include_vertices', action="store_true", default=False,
                        help='include vertices in the result file')
    parser.add_argument('--output_format', type=str, default='DEBUG',
                        help='format for packing the resultr: LWGAN or DEBUG')
    parser.add_argument('--trim_frames', action="store_true", default=True,
                        help='trim excess frames to synch different cameras')
    parser.add_argument('--pose_by_rotmat', action="store_true",
                        help='fix a pose by rotmat')
    args = parser.parse_args()

    args.include_vertices = True
    args.output_format = 'LWGAN'
    args.smpl_dir = 'smpls_by_vibe'
    args.result_dir = 'smpls_by_vibe_aligned_lwgan_fix'
    args.pose_by_rotmat = True

    align_smpl(args)

if __name__ == '__main__':
    main()