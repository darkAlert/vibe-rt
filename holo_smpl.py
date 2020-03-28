import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import torch
import joblib
import argparse
import numpy as np
from torch.utils.data import DataLoader
from lib.models.vibe import VIBE_Demo
from lib.dataset.inference import HoloInference
from lib.utils.demo_utils import convert_crop_cam_to_orig_img
from holo.data_struct import DataStruct
from lib.utils.renderer import Renderer
import colorsys


def predict_smpl(args, debug_render=False):
    # Get data struct:
    output_dir = os.path.join(args.root_dir, args.output_dir)
    frames_dir = os.path.join(args.root_dir, args.frames_dir)
    bboxes_dir = os.path.join(args.root_dir, args.bboxes_dir)
    data_frames = DataStruct().parse(frames_dir, levels='subject/light/garment/scene/cam', ext='jpeg')
    data_bboxes = DataStruct().parse(bboxes_dir, levels='subject/light/garment/scene/cam', ext='npz')

    # Init VIBE model:
    assert os.path.exists(args.vibe_model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vibe_model = VIBE_Demo(seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True).to(device)
    ckpt = torch.load(args.vibe_model_path)
    vibe_model.load_state_dict(ckpt['gen_state_dict'], strict=False)
    vibe_model.eval()
    print('Loaded pretrained weights from', args.vibe_model_path)
    print('Performance of pretrained model on 3DPW:', ckpt["performance"])

    # Init rendered if needed:
    if debug_render is True:
        renderer = Renderer(resolution=(1920, 1080), orig_img=True, wireframe=True, gender='male')
        mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

    # Run VIBE model for all persons:
    for (f_node, f_path), (b_node, b_path) in zip(data_frames.nodes('cam'), data_bboxes.nodes('cam')):
        if args.person is not None:
            person = f_path.split('/')[0]
            if person != args.person:
                continue
        if args.filter_by_path is not None:
            cur_path = f_path.split('/cam')[0]
            if cur_path != args.filter_by_path:
                continue
        print ('Processing dir', f_path)
        assert f_path == b_path

        # Unpack npz containing bboxes:
        bboxes_path = [npz.abs_path for npz in data_bboxes.items(b_node)][0]
        bboxes_npz = np.load(bboxes_path, encoding='latin1', allow_pickle=True)
        frame_ids = np.array(bboxes_npz['frames'])
        bboxes = np.array(bboxes_npz['bboxes'])
        bboxes[:,0] = bboxes[:,0] + bboxes[:,2]*0.5     # (x,y,w,h) -> (cx,cy,w,h)
        bboxes[:,1] = bboxes[:,1] + bboxes[:,3]*0.5

        # Prepare frames:
        frame_paths = np.array([f.path for f in data_frames.items(f_node)])
        frame_paths = frame_paths[frame_ids]

        # Make pytorch dataloader:
        dataset = HoloInference(root_dir=frames_dir, frame_paths=frame_paths, bboxes=bboxes, scale=args.bbox_scale)
        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=args.num_workers)

        # Predict SMPL using VIBE model:
        with torch.no_grad():
            pred_cam, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], []
            pred_verts = []
            pred_joints2d, rotmat = [], []

            for batch in dataloader:
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]
                output = vibe_model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                if args.add_verts:
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:, :, 3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :, 75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                pred_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))
                rotmat.append(output['rotmat'].reshape(batch_size * seqlen, -1, 3, 3))

            pred_cam = torch.cat(pred_cam, dim=0).cpu().numpy()
            if args.add_verts:
                pred_verts = torch.cat(pred_verts, dim=0).cpu().numpy()
            pred_pose = torch.cat(pred_pose, dim=0).cpu().numpy()
            pred_betas = torch.cat(pred_betas, dim=0).cpu().numpy()
            pred_joints3d = torch.cat(pred_joints3d, dim=0).cpu().numpy()
            pred_joints2d = torch.cat(pred_joints2d, dim=0).cpu().numpy()
            rotmat = torch.cat(rotmat, dim=0).cpu().numpy()

        # Prepare the result:
        path = os.path.join(frames_dir, frame_paths[0])
        h, w = cv2.imread(path).shape[:2]
        orig_cam = convert_crop_cam_to_orig_img(cam=pred_cam, bbox=bboxes, img_width=w, img_height=h)

        if args.add_verts:
            vibe_result = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'n_joints2d': pred_joints2d,
                'rotmat': rotmat,
                'bboxes': bboxes,
                'frame_paths': frame_paths
            }
        else:
            vibe_result = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'n_joints2d': pred_joints2d,
                'rotmat': rotmat,
                'bboxes': bboxes,
                'frame_paths': frame_paths
            }

        # Renderer the result:
        if debug_render is True:
            idx = 0
            img_path = os.path.join(frames_dir, vibe_result['frame_paths'][idx])
            img = cv2.imread(img_path)
            pred_verts = vibe_result['verts'][idx]
            orig_cam = vibe_result['orig_cam'][idx]
            result_img = renderer.render(img, pred_verts, cam=orig_cam, color=mesh_color)
            cv2.imshow('result_img', result_img)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        # Save the result:
        else:
            result_dir = os.path.join(output_dir, f_path)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_path = os.path.join(result_dir, 'smpl.pkl')
            joblib.dump(vibe_result, result_path)

    print ('All done!')


def main():
    # Set VIBE params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/darkalert/KazendiJob/Data/HoloVideo/Data',
                        help='root dir path')
    parser.add_argument('--frames_dir', type=str, default='frames',
                        help='path to dir with frames relatively to the root_dir')
    parser.add_argument('--bboxes_dir', type=str, default='bboxes_by_maskrcnn',
                        help='path to dir with bounding boxes relatively to the root_dir')
    parser.add_argument('--output_dir', type=str, default='smpls',
                        help='output folder to write results')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of workers for dataloader')
    parser.add_argument('--vibe_batch_size', type=int, default=64,
                        help='batch size of VIBE')
    parser.add_argument('--vibe_model_path', type=str, default='data/vibe_data/vibe_model_wo_3dpw.pth.tar',
                        help='path to pretrained VIBE model')
    parser.add_argument('--bbox_scale', type=int, default=1.1,
                        help='scale for bounding boxes')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu id')
    parser.add_argument('--person', type=str, default=None,
                        help='filter data by person')
    parser.add_argument('--filter_by_path', type=str, default=None,
                        help='filter data by given path')
    parser.add_argument('--add_verts', action='store_true',
                        help='add vertices to output npz')
    parser.add_argument('--debug', action='store_false',
                        help='')
    args = parser.parse_args()

    # Params:
    if args.debug:
        args.frames_dir = 'frames'
        # args.bboxes_dir = 'bboxes'
        args.bboxes_dir = 'bboxes_by_maskrcnn'
        args.output_dir = 'smpl' if args.bboxes_dir == 'bboxes' else 'smpl_maskrcnn2'
        args.bbox_scale = 1.0 if args.bboxes_dir == 'bboxes' else 1.1
        args.filter_by_path = 'person_2/light-100_temp-5600/garments_2/front_position'
        args.add_verts=True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('person:', args.person, ', gpu_id:', args.gpu_id, ', root_dir:', args.root_dir)
    if args.filter_by_path is not None:
        print ('filter by path:', args.filter_by_path)

    predict_smpl(args, debug_render=False)

if __name__ == '__main__':
    main()


