import os

from tqdm import tqdm


def jpeg_to_jpg(path):
    for root, dirs, files in tqdm(list(os.walk(path))):
        for fname in files:
            if fname.endswith(".jpeg"):
                os.rename(os.path.join(root, fname), os.path.join(root, fname.replace(".jpeg", ".jpg")))


def check_frames(path):
    for root, dirs, files in tqdm(list(os.walk(path))):
        if set(dirs) in set(f"cam{i}" for i in range(1, 7)):
            for i in range(1, 7):
                if not os.path.exists(os.path.join(root, f"cam{i}")):
                    print(f"{root}: cam{i} is missing")

        if os.path.basename(root) in [f"cam{i}" for i in range(1, 7)]:
            if len(files) == 0:
                print(f"{root} is empty")

            last_frame_num = int(max(files).split(".")[0].split("-")[1])

            for i in range(1, last_frame_num + 1):
                if not os.path.exists(os.path.join(root, f"image-{str(i).zfill(5)}.jpg")):
                    print(f"{root}: frame {i} is missing")


if __name__ == '__main__':
    check_frames(r"/home/kazendi/vlad/data/HoloVideo")

