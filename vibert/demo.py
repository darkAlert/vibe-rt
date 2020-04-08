# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
# from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    frames_dir = os.path.join(args.root_dir, args.frames_dir)

    if not os.path.exists(frames_dir):
        exit(f'Input video \"{frames_dir}\" does not exist!')

    image_folder = frames_dir
    num_frames = len(os.listdir(frames_dir))
    img_shape = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0])).shape

    # image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    if args.tracking_method == 'pose':
        raise ValueError("openpose is available for video only")
        # if not os.path.isabs(video_file):
        #     video_file = os.path.join(os.getcwd(), video_file)
        # tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=True) # args.display
    elif args.tracking_method == "bbox":
        # run multi object tracker
        raise ValueError("Yolo is not available now")
        # mot = MPT(
        #     device=device,
        #     batch_size=args.tracker_batch_size,
        #     display=args.display,
        #     detector_type=args.detector,
        #     output_format='dict',
        #     yolo_img_size=args.yolo_img_size,
        # )
        # tracking_results = mot(image_folder)
    elif os.path.exists(args.tracking_method):
        boxes_xyxy = np.load(args.tracking_method)["boxes"]
        boxes_xywh = np.zeros_like(boxes_xyxy)

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        tracking_results = {"1": {"bbox": boxes_xywh,
                                  "frames": np.array(range(boxes_xywh.shape[0]))}}
    else:
        raise ValueError(f"{args.tracking_method} not in ['pose', 'bbox'] and such path doesn't exist")

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
        # use_6d=False, # for alternative prediction, doesn't work(((
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if args.tracking_method == 'bbox' or os.path.exists(args.tracking_method):
            bboxes = tracking_results[person_id]['bbox']
        elif args.tracking_method == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=12)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []
            pred_joints2d, rotmat = [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                pred_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))
                rotmat.append(output['rotmat'].reshape(batch_size * seqlen, -1, 3, 3))  


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            
            pred_joints2d = torch.cat(pred_joints2d, dim=0)
            rotmat = torch.cat(rotmat, dim=0)

            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args.run_smplify and args.tracking_method == 'pose':
            # TODO: will not work with non-openpose keypoints
           norm_joints2d = np.concatenate(norm_joints2d, axis=0)
           norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
           norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

           # Run Temporal SMPLify
           update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
           new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
               pred_rotmat=pred_pose,
               pred_betas=pred_betas,
               pred_cam=pred_cam,
               j2d=norm_joints2d,
               device=device,
               batch_size=norm_joints2d.shape[0],
               pose2aa=False,
           )

           # update the parameters after refinement
           print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
           pred_verts = pred_verts.cpu()
           pred_cam = pred_cam.cpu()
           pred_pose = pred_pose.cpu()
           pred_betas = pred_betas.cpu()
           pred_verts[update] = new_opt_vertices[update]
           pred_cam[update] = new_opt_cam[update]
           pred_pose[update] = new_opt_pose[update]
           pred_betas[update] = new_opt_betas[update]
           pred_joints3d[update] = new_opt_joints3d[update].cuda()

        elif args.run_smplify and args.tracking_method == 'bbox':
           print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
           print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        pred_joints2d = pred_joints2d.cpu().numpy()
        rotmat = rotmat.cpu().numpy()
 
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        if os.path.exists(args.tracking_method):
            kp = np.load(args.tracking_method)["keypoints"]
            joints2d = {"0": {"joints2d": kp,
                              "frames": np.array(range(kp.shape[0]))}}

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'n_joints2d': pred_joints2d,
            'rotmat': rotmat,            
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    # making output to the same directory
    output_dir = os.path.join(args.output_dir, args.frames_dir)
    os.makedirs(output_dir, exist_ok=True)

    result_name = os.path.join(output_dir, "result.pkl")
    print(f'Saving output results to \"{result_name}\".')

    joblib.dump(vibe_results, result_name)

    if False:
    # if not args.no_render:  # TODO: might not work due to paths
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                )

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:05d}.jpg'), img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        # #todo: change directories
        # vid_name = os.path.basename(video_file)
        # save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
        # #save_name = os.path.join(output_path, save_name)
        # save_name = os.path.join(args.output_dir, save_name)
        # print(f'Saving result video to {save_name}')
        # images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        # shutil.rmtree(output_img_folder)

    # timely off it cause will use separate prediction and rendering
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # python demo1.py --root_dir=/home/kazendi/vlad/data/HoloVideo/frames --frames_dir=person_1/light-100_temp-5600/garment_1/freestyle/cam1 --output_dir=/home/kazendi/vlad/data/HoloVideo/frames_res --wireframe --no_render

    parser.add_argument('--root_dir', type=str,
                        help='root dir path')

    parser.add_argument('--frames_dir', type=str,
                        help='path to dir with frames relatively to the root_dir')

    parser.add_argument('--output_dir', type=str,
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox',
                        help='tracking method to calculate the tracklet of a subject from the input video. '
                             'Either bbox or path to precalculated boxes/joints.')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes')

    args = parser.parse_args()

    # Params:
    args.root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/'
    args.frames_dir = 'frames/person_1/light-100_temp-5600/garment_1/freestyle/cam1'
    args.tracking_method = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/poses/person_1/light-100_temp-5600/garment_1/freestyle/cam1/cam1.npz'

    main(args)

