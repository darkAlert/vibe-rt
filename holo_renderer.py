import os
import cv2
import joblib
import argparse
import numpy as np
from holo.data_struct import DataStruct
from lib.utils.renderer import Renderer
import colorsys


def render_smpl(args):
    # Parse params:
    root_dir = args.root_dir
    smpl_dir = args.smpl_dir
    frames_dir = args.frames_dir
    output_dir = args.output_dir
    width = args.width
    height = args.height
    gender = args.gender
    format = args.input_format
    render_cam = args.cam

    # Load data:
    output_dir = os.path.join(root_dir, output_dir)
    frames_dir = os.path.join(root_dir, frames_dir)
    smpl_dir = os.path.join(root_dir, smpl_dir)
    data_smpl = DataStruct().parse(smpl_dir, levels='subject/light/garment/scene/cam', ext=format)

    # Init renderer:
    renderer = Renderer(resolution=(width, height), orig_img=True, wireframe=True, gender=gender)
    mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

    # Render:
    for node, path in data_smpl.nodes('cam'):
        print ('Processing dir', path)

        smpl_path = [smpl.abs_path for smpl in data_smpl.items(node)][0]
        if format == 'pkl':
            vibe_result = joblib.load(smpl_path, 'r')
        elif format == 'npz':
            vibe_result = np.load(smpl_path, encoding='latin1', allow_pickle=True)
        else:
            raise NotImplementedError

        n = len(vibe_result['frame_paths'])
        if args.max_frames > 0:
            n = min(n, args.max_frames)

        result_dir = os.path.join(output_dir,path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for idx in range(n):
            img_path = os.path.join(frames_dir, vibe_result['frame_paths'][idx])
            img = cv2.imread(img_path)
            pred_verts = vibe_result['verts'][idx]
            cam = vibe_result[render_cam][idx]
            result_img = renderer.render(img, pred_verts, cam=cam,color=mesh_color)

            #Save:
            name = vibe_result['frame_paths'][idx].split('/')[-1]
            out_path = os.path.join(result_dir, name)
            cv2.imwrite(out_path, result_img)

            print('{}/{}      '.format(idx + 1, n), end='\r')

    print ('All done!')


def main():
    # Set renderer params:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/darkalert/KazendiJob/Data/HoloVideo/Data',
                        help='root dir path')
    parser.add_argument('--frames_dir', type=str, default='frames',
                        help='path to dir with frames relatively to the root_dir')
    parser.add_argument('--smpl_dir', type=str, default='smpl_maskrcnn',
                        help='path to dir with smpls relatively to the root_dir')
    parser.add_argument('--output_dir', type=str, default='rendered_smpl_maskrcnn',
                        help='output folder to write results')
    parser.add_argument('--width', type=int, default=1920,
                        help='width of the result image')
    parser.add_argument('--height', type=int, default=1080,
                        help='height of the result image')
    parser.add_argument('--gender', type=str, default='male',
                        help='gender of smpl for rendering')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='max number of frames for rendering')
    parser.add_argument('--input_format', type=str, default='npz',
                        help='npz or pkl')
    parser.add_argument('--cam', type=str, default='orig_cam',
                        help='cam for rendering: orig_cam, pred_cam or avatar_cam')
    args = parser.parse_args()

    args.smpl_dir = 'smpl_maskrcnn_aligned'
    args.output_dir = 'rendered_smpl_maskrcnn_aligned'
    args.cam = 'avatar_cam'
    args.width = 256
    args.height = 256
    args.max_frames = 250

    render_smpl(args)


if __name__ == '__main__':
    main()