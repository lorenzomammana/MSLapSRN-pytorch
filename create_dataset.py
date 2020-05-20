import glob
import cv2
from tqdm import tqdm


def generate_patches(outfolder, stride):
    images = [f for f in glob.glob("dataset/{}/*.png".format(outfolder))]

    with open("{}_patches.txt".format(outfolder), "w+") as f:
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
                    cv2.imwrite("dataset/{}_patches/lr/{}_{}_{}.png".format(outfolder, outname, i, j), patch_lr)
                    cv2.imwrite("dataset/{}_patches/2x/{}_{}_{}.png".format(outfolder, outname, i, j), patch_2x)
                    cv2.imwrite("dataset/{}_patches/4x/{}_{}_{}.png".format(outfolder, outname, i, j), patch_4x)


if __name__ == '__main__':
    size_label = 128
    scale = 4

    size_input = size_label / scale
    size_x2 = size_label / 2

    generate_patches('train', stride=15)
    generate_patches('validation', stride=30)


