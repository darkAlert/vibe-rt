import os
import sys
import subprocess

def main():
  src_path = sys.argv[1]
  print('Processing SMPL multiview from this folder:', src_path)
  NUM_CAMS = 6

  print("======================STAGE I NORMALIZING VIDEOS to 30 fps=====================")
  for i in range(NUM_CAMS):
      cmd = [
          'python',
          'norm_video_fps.py',
          src_path
      ]
      print('Executing', ' '.join(cmd))
      subprocess.call(cmd)

  print("======================STAGE II CALCULATE VIBE PREDICT FOR EACH VIDEO======================")
  # python demo.py --vid_file hp_chrkey_data/p4_fs_l100/norm_30fps/cam1.mp4 --tracking_method pose --staf_dir ./STAF/openpose  --output_folder hp_norm_smpl_labels/ --wireframe --no_render --run_smplify
  for i in range(NUM_CAMS):
      video_fname = src_path + 'cam' + str(i+1) + ".mp4"
      cmd = [
          'python',
          'demo.py',
          '--vid_file', video_fname,
          '--tracking_method', 'pose',
          '--staf_dir', './STAF/openpose',
          '--output_folder', src_path + 'smpl_align',
          '--wireframe',
          '--no_render',
          '--run_smplify'
      ]
      print('Executing', ' '.join(cmd))
      subprocess.call(cmd)

  print("======================STAGE III CALCULATE SMPL AVERAGE FROM PREDICTS======================")
  predict_src = src_path + '/smpl_align'
  cmd = [
        'python',
        'smpl_align.py',
         predict_src
  ]
  print('Executing', ' '.join(cmd))
  subprocess.call(cmd)

  print("======================STAGE IV FINAL RENDERING FROM AVERAGED PREDICT======================")
  #python demo_render.py --vid_file hp_chrkey_data/p4_fs_l100/norm_30fps/cam1.mp4 --vibe_predict_file hp_norm_smpl_labels/cam1_fin.pkl --output_folder smpl_align/ --wireframe
  for i in range(NUM_CAMS):
      video_fname = src_path + '/norm_30fps/cam' + str(i + 1) + ".mp4"
      predict_fname = src_path + '/smpl_align/cam' + str(i + 1) + "_fin.pkl"
      output_path = src_path + '/smpl_align'
      cmd = [
          'python',
          'demo_render.py',
          '--vid_file', video_fname,
          '--vibe_predict_file', predict_fname,
          '--output_folder', output_path,
          '--wireframe'
      ]
      print('Executing', ' '.join(cmd))
      subprocess.call(cmd)

  print('================= FINISH =================')

if __name__ == '__main__':
  main()