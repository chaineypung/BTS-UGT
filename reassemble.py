import numpy as np
import glob
import os
import nibabel as nib
import cv2
import scipy.ndimage
from tqdm import tqdm

def resize(img, shape, mode='constant', orig_shape=None):

    assert len(shape) == 3, "Can not have more than 3 dimensions"

    factors = (
        shape[0] / orig_shape[0],
        shape[1] / orig_shape[1],
        shape[2] / orig_shape[2]
    )
    return scipy.ndimage.zoom(img, factors, mode=mode)


def preprocess_label(img, out_shape=None, mode='nearest'):

    ncr = img == 1
    ed = img == 2
    et = img == 4

    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode, orig_shape=img.shape)
        ed = resize(ed, out_shape, mode=mode, orig_shape=img.shape)
        et = resize(et, out_shape, mode=mode, orig_shape=img.shape)

    ncr = np.where(ncr > 0, 1, 0)
    ed = np.where(ed > 0, 2, 0)
    et = np.where(et > 0, 4, 0)
    Img = ncr + ed + et
    return np.array(Img, dtype=np.uint8)


def ext_roi(path, out_path):
    dirPath = glob.iglob(path)
    for big_file in dirPath:
        files = os.listdir(big_file)
        for file in files:
            Mask = nib.load(big_file + '/' + file + '/' + file + '_seg.nii.gz')# one stage mask
            T1 = nib.load(big_file + '/' + file + '/' + file + '_t1.nii.gz')
            T1ce = nib.load(big_file + '/' + file + '/' + file + '_t1ce.nii.gz')
            T2 = nib.load(big_file + '/' + file + '/' + file + '_t2.nii.gz')
            Flair = nib.load(big_file + '/' + file + '/' + file + '_flair.nii.gz')
            Mask1 = nib.load('E:/pcx/BraTS2020-2stage/MICCAI_BraTS2020_ValidationData/' + file + '/' + file + '_pre.nii.gz')# two stage mask

            mask = np.float32(Mask.dataobj)# For uncertainty, float32
            mask1 = np.float32(Mask1.dataobj)# For uncertainty, float32
            mask_final = np.zeros_like(mask)
            t1 = np.float32(T1.dataobj)
            t1ce = np.float32(T1ce.dataobj)
            t2 = np.float32(T2.dataobj)
            flair = np.float32(Flair.dataobj)

            all_input_image = np.stack([t1, t1ce, t2, flair], axis=0)

            x, y, z = [], [], []
            for i in range(0, mask.shape[0], 1):
                num = cv2.countNonZero(mask[i, :, :])
                if num != 0:
                    x.append(i)
            for j in range(0, mask.shape[1], 1):
                num = cv2.countNonZero(mask[:, j, :])
                if num != 0:
                    y.append(j)
            for k in range(0, mask.shape[2], 1):
                num = cv2.countNonZero(mask[:, :, k])
                if num != 0:
                    z.append(k)

            x_min = np.array(x).min()
            x_max = np.array(x).max()
            y_min = np.array(y).min()
            y_max = np.array(y).max()
            z_min = np.array(z).min()
            z_max = np.array(z).max()

            x_indexes, y_indexes, z_indexes = np.nonzero(np.sum(all_input_image, axis=0) != 0)
            xmin, ymin, zmin = [max(0, int(np.min(arr) - 1)) for arr in (x_indexes, y_indexes, z_indexes)]
            xmax, ymax, zmax = [int(np.max(arr) + 1) for arr in (x_indexes, y_indexes, z_indexes)]

            x_cut_min = int(x_min - 20)
            x_cut_max = int(x_max + 20)
            y_cut_min = int(y_min - 20)
            y_cut_max = int(y_max + 20)
            z_cut_min = int(z_min - 10)
            z_cut_max = int(z_max + 10)

            if x_cut_min < xmin:
                x_cut_min = xmin
            if y_cut_min < ymin:
                y_cut_min = ymin
            if z_cut_min < zmin:
                z_cut_min = zmin
            if x_cut_max > xmax:
                x_cut_max = xmax
            if y_cut_max > ymax:
                y_cut_max = ymax
            if z_cut_max > zmax:
                z_cut_max = zmax

            mask_final[x_cut_min:x_cut_max, y_cut_min:y_cut_max, z_cut_min:z_cut_max] = mask1

            # print((mask == mask_final).all())

            mask_final = nib.Nifti1Image(mask_final, Mask.affine, Mask.header)

            # if not (os.path.exists(out_path)):
            #     os.mkdir(out_path)
            # if not (os.path.exists(out_path + '/' + file)):
            #     os.mkdir(out_path + '/' + file)

            nib.save(mask_final, out_path + '/' + file + '.nii.gz')
            print(f"{file} done!")


if __name__=='__main__':
    ext_roi(path=r'E:/pcx/BraTS2020/MICCAI_BraTS2020_ValidationData',
            out_path=r'E:/pcx/submission')