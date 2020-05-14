import glob
import cv2
from tqdm import tqdm

size_label = 128
scale = 4

size_input = size_label / scale
size_x2 = size_label / 2
stride = 20

images = [f for f in glob.glob("dataset/train/*.png")]

with open("train_patches.txt", "w+") as f:
    for imagename in tqdm(images):
        img = cv2.imread(imagename)

        for i in range(0, img.shape[0] - size_label, stride):
            for j in range(0, img.shape[1] - size_label, stride):
                patch_4x = img[i:i + size_label, j:j + size_label, :]
                patch_lr = cv2.resize(patch_4x, dsize=(int(size_label / 4), int(size_label / 4)),
                                      interpolation=cv2.INTER_CUBIC)
                patch_2x = cv2.resize(patch_4x, dsize=(int(size_label / 2), int(size_label / 2)),
                                      interpolation=cv2.INTER_CUBIC)

                outname = imagename.split("/")[-1].split(".")[0]

                f.write("{}_{}_{}.png\n".format(outname, i, j))
                cv2.imwrite("dataset/train_patches/lr/{}_{}_{}.png".format(outname, i, j), patch_lr)
                cv2.imwrite("dataset/train_patches/2x/{}_{}_{}.png".format(outname, i, j), patch_2x)
                cv2.imwrite("dataset/train_patches/4x/{}_{}_{}.png".format(outname, i, j), patch_4x)
