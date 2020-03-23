import os
import joblib
import numpy as np
from data_struct import DataStruct

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

def convert_crop_cam_to_another_crop(cam, bbox1, bbox2, img_width, img_height):
    bbox = bbox1[:,:3] - bbox2[:,:3]
    bbox[:,2] = bbox1[:,2]
    img_width = bbox2[0,2]
    img_height = bbox2[0,3]

    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx ) / hw / sx) + cam[:,1]
    ty = ((cy ) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def recalculate_avatar_camera_params(smpl_dir, bboxes_dir, output_dir, width=1920, height=1080):
    data_smpl = DataStruct().parse(smpl_dir, levels='subject/light/garment/scene/cam', ext='pkl')
    data_bbox = DataStruct().parse(bboxes_dir, levels='subject/light/garment/scene/cam', ext='npz')

    for (s_node, s_path), (b_node, b_path) in zip(data_smpl.nodes('cam'), data_bbox.nodes('cam')):
        print('Processing dir', s_path)
        assert s_path == b_path, (s_path, b_path)

        # Unpack a npz containing bboxes:
        bboxes_path = [npz.abs_path for npz in data_bbox.items(b_node)][0]
        bboxes_npz = np.load(bboxes_path, encoding='latin1', allow_pickle=True)
        bboxes = np.array(bboxes_npz['bboxes'])
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5  # (x,y,w,h) -> (cx,cy,w,h)
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5

        # Unpack a pickle containing smpl params:
        smpl_path = [smpl.abs_path for smpl in data_smpl.items(s_node)][0]
        vibe_result = joblib.load(smpl_path, 'r')
        assert len(vibe_result['frame_paths']) == bboxes.shape[0], (len(vibe_result['frame_paths']), bboxes.shape[0])

        # Recalculate cam params:
        pred_cam = vibe_result['pred_cam']
        orig_bboxes = vibe_result['bboxes']
        avatar_cam = convert_crop_cam_to_another_crop(cam=pred_cam,
                                                      bbox1=orig_bboxes,
                                                      bbox2=bboxes,
                                                      img_width=width, img_height=height)

        # Save:
        result_dir= os.path.join(output_dir, s_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'smpl.npz')
        np.savez(result_path,
                 avatar_cam=avatar_cam,
                 avatar_bboxes=bboxes,
                 pred_cam=vibe_result['pred_cam'],
                 orig_cam=vibe_result['orig_cam'],
                 pose=vibe_result['pose'],
                 betas=vibe_result['betas'],
                 joints3d=vibe_result['joints3d'],
                 n_joints2d=vibe_result['n_joints2d'],
                 rotmat=vibe_result['rotmat'],
                 bboxes=vibe_result['bboxes'],
                 frame_paths=vibe_result['frame_paths'])

    print('All done!')


def main():
    smpl_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpl_maskrcnn'
    bboxes_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/bboxes'
    output_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/smpl_maskrcnn_fixed'
    recalculate_avatar_camera_params(smpl_dir, bboxes_dir, output_dir)

if __name__ == '__main__':
    main()