import os
import shutil

for root, dirs, files in os.walk("/home/kazendi/vlad/data/mp4"):
    for f in files:
        if f not in [f"cam{i}.mp4" for i in range(1, 7)]:
            print(os.path.join(root, f))
            os.remove(os.path.join(root, f))

for root, dirs, files in os.walk("/home/kazendi/vlad/data/mp4"):
    for d in dirs:
        if d in ["smpl_align", "norm_30fps"]:
            print(os.path.join(root, d))
            shutil.rmtree(os.path.join(root, d))

