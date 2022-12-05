import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset

from config import get_brats_folder, get_test_brats_folder
from dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise


# def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
#     """
#     Only do intensity shift on brain voxels
#     :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
#     :param brain_mask:
#     :param limit:
#     :return:
#     """
#
#     shift_range = 2 * limit
#     for i in range(len(imgs_array) - 1):
#         factor = -limit + shift_range * np.random.random()
#         std = imgs_array[i][brain_mask].std()
#         imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
#     return imgs_array
#
#
# def random_scale(imgs_array, brain_mask, scale_limits=(0.9, 1.1)):
#     """
#     Only do random_scale on brain voxels
#     :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
#     :param scale_limits:
#     :return:
#     """
#     scale_range = scale_limits[1] - scale_limits[0]
#     for i in range(len(imgs_array) - 1):
#         factor = scale_limits[0] + scale_range * np.random.random()
#         imgs_array[i][brain_mask] = imgs_array[i][brain_mask] * factor
#     return imgs_array
#
#
# def random_mirror_flip(imgs_array, labels_array, prob=0.5):
#     """
#     Perform flip along each axis with the given probability; Do it for all voxels；
#     labels should also be flipped along the same axis.
#     :param imgs_array:
#     :param prob:
#     :return:
#     """
#     imglabel_array = np.concatenate([imgs_array, labels_array], axis=0)
#     for axis in range(1, len(imglabel_array.shape)):
#         random_num = np.random.random()
#         if random_num >= prob:
#             if axis == 1:
#                 imglabel_array = imglabel_array[:, ::-1, :, :]
#             if axis == 2:
#                 imglabel_array = imglabel_array[:, :, ::-1, :]
#             if axis == 3:
#                 imglabel_array = imglabel_array[:, :, :, ::-1]
#     imgs_array = imglabel_array[:4, ...]
#     labels_array = imglabel_array[4:, ...]
#     return imgs_array, labels_array
#
#
#
# class Brats(Dataset):
#     def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
#                  no_seg=False, normalisation="minmax"):
#         super(Brats, self).__init__()
#         self.benchmarking = benchmarking
#         self.normalisation = normalisation
#         self.data_aug = data_aug
#         self.training = training
#         self.datas = []
#         self.validation = no_seg
#         if self.training:
#             self.patterns = ["_t1", "_t1ce", "_t2", "_flair", "_uncertainty"]
#         else:
#             self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
#         if not no_seg:
#             self.patterns += ["_seg"]
#         for patient_dir in patients_dir:
#             patient_id = patient_dir.name
#             paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
#             if self.training:
#                 patient = dict(
#                     id=patient_id, t1=paths[0], t1ce=paths[1],
#                     t2=paths[2], flair=paths[3], seg=paths[4], uncertainty=paths[5] if not no_seg else None
#                 )
#             else:
#                 patient = dict(
#                     id=patient_id, t1=paths[0], t1ce=paths[1],
#                     t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
#                 )
#             self.datas.append(patient)
#
#     def __getitem__(self, idx):
#         _patient = self.datas[idx]
#         if self.training:
#             patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg", "uncertainty"]}
#         else:
#             patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
#         if _patient["seg"] is not None:
#             patient_label = self.load_nii(_patient["seg"])
#         if self.normalisation == "minmax":
#             patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
#         elif self.normalisation == "zscore":
#             patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
#         patient_image = np.stack([patient_image[key] for key in patient_image])
#
#         if _patient["seg"] is not None:
#             et = patient_label == 4
#             et_present = 1 if np.sum(et) >= 1 else 0
#             tc = np.logical_or(patient_label == 4, patient_label == 1)
#             wt = np.logical_or(tc, patient_label == 2)
#             patient_label = np.stack([et, tc, wt])
#         else:
#             patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
#             et_present = 0
#         if self.training:
#             # Remove maximum extent of the zero-background to make future crop more useful
#             z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
#             # Add 1 pixel in each side
#             zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
#             zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
#             patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
#             patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
#
#             uncer = self.load_nii(_patient["uncertainty"])
#             uncer = uncer[zmin:zmax, ymin:ymax, xmin:xmax]
#             uncer = np.expand_dims(uncer, axis=0)
#             patient_image = np.concatenate((patient_image, uncer), axis=0)
#
#
#
#             # add data aug code...
#             # mask = patient_label[0, ...] + patient_label[1, ...] + patient_label[2, ...]
#             # mask = mask != 0
#             # patient_image = random_intensity_shift(patient_image, mask)
#             # patient_image = random_scale(patient_image, mask)
#
#             # default to 128, 128, 128 64, 64, 64 32, 32, 32
#             patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128)) # (128, 128, 128)
#
#             # add data aug code...
#             # patient_image, patient_label = random_mirror_flip(patient_image, patient_label)
#         else:
#             z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
#             # Add 1 pixel in each side
#             zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
#             zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
#             patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
#             patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
#
#         patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
#         patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
#         return dict(patient_id=_patient["id"],
#                     image=patient_image, label=patient_label,
#                     seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
#                     crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
#                     et_present=et_present,
#                     supervised=True,
#                     )
#
#     @staticmethod
#     def load_nii(path_folder):
#         return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))
#
#     def __len__(self):
#         return len(self.datas)
#
#
# def get_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
#     # base_folder = pathlib.Path(get_brats_folder(on)).resolve()
#     # print(base_folder)
#     # assert base_folder.exists()
#     # patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
#     #
#     # kfold = KFold(3, shuffle=True, random_state=seed)
#     # splits = list(kfold.split(patients_dir))
#     # train_idx, val_idx = splits[fold_number]
#     # len_val = len(val_idx)
#     # val_index = val_idx[: len_val//2]
#     # test_index = val_idx[len_val // 2 :]
#     #
#     # train = [patients_dir[i] for i in train_idx]
#     # val = [patients_dir[i] for i in val_index]
#     # test = [patients_dir[i] for i in test_index]
#
#     train_folder = pathlib.Path(get_brats_folder("train")).resolve()
#     valid_folder = pathlib.Path(get_brats_folder("val")).resolve()
#     test_folder = pathlib.Path(get_brats_folder("test")).resolve()
#     # print(base_folder)
#     # assert base_folder.exists()
#     train_dir = sorted([x for x in train_folder.iterdir() if x.is_dir()])
#     valid_dir = sorted([x for x in valid_folder.iterdir() if x.is_dir()])
#     test_dir = sorted([x for x in test_folder.iterdir() if x.is_dir()])
#
#     # kfold = KFold(3, shuffle=True, random_state=seed)
#     # splits = list(kfold.split(patients_dir))
#     # train_idx, val_idx = splits[fold_number]
#     # len_val = len(val_idx)
#     # val_index = val_idx[: len_val // 2]
#     # test_index = val_idx[len_val // 2:]
#
#     train = train_dir
#     val = valid_dir
#     test = test_dir
#
#     # return patients_dir
#     train_dataset = Brats(train, training=True,
#                           normalisation=normalisation)
#     val_dataset = Brats(val, training=False, data_aug=False,
#                         normalisation=normalisation)
#     bench_dataset = Brats(test, training=False, benchmarking=True,
#                           normalisation=normalisation)
#     return train_dataset, val_dataset, bench_dataset
#
#
# def get_test_datasets(seed, on="train", fold_number=0, normalisation="zscore"):
#     base_folder = pathlib.Path(get_test_brats_folder()).resolve()
#     print(base_folder)
#     assert base_folder.exists()
#     patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
#
#     bench_dataset = Brats(patients_dir, training=False, benchmarking=True,
#                           normalisation=normalisation)
#     return bench_dataset

def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
    """
    Only do intensity shift on brain voxels
    :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
    :param brain_mask:
    :param limit:
    :return:
    """

    shift_range = 2 * limit
    for i in range(len(imgs_array) - 1):
        factor = -limit + shift_range * np.random.random()
        std = imgs_array[i][brain_mask].std()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
    return imgs_array


def random_scale(imgs_array, brain_mask, scale_limits=(0.9, 1.1)):
    """
    Only do random_scale on brain voxels
    :param imgs_array: The whole input image with shape of (4, 155, 240, 240)
    :param scale_limits:
    :return:
    """
    scale_range = scale_limits[1] - scale_limits[0]
    for i in range(len(imgs_array) - 1):
        factor = scale_limits[0] + scale_range * np.random.random()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] * factor
    return imgs_array


def random_mirror_flip(imgs_array, labels_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels；
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    imglabel_array = np.concatenate([imgs_array, labels_array], axis=0)
    for axis in range(1, len(imglabel_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imglabel_array = imglabel_array[:, ::-1, :, :]
            if axis == 2:
                imglabel_array = imglabel_array[:, :, ::-1, :]
            if axis == 3:
                imglabel_array = imglabel_array[:, :, :, ::-1]
    imgs_array = imglabel_array[:4, ...]
    labels_array = imglabel_array[4:, ...]
    return imgs_array, labels_array



class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="zscore"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

            # uncer = self.load_nii(f"E:/pcx/MICCAI2018_2stage/train/{_patient['id']}/{_patient['id']}_uncertainty.nii.gz")
            # uncer = uncer[zmin:zmax, ymin:ymax, xmin:xmax]
            # uncer = np.expand_dims(uncer, axis=0)
            # patient_image = np.concatenate((patient_image, uncer), axis=0)

            # add data aug code...
            # mask = patient_label[0, ...] + patient_label[1, ...] + patient_label[2, ...]
            # mask = mask != 0
            # patient_image = random_intensity_shift(patient_image, mask)
            # patient_image = random_scale(patient_image, mask)

            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128)) # (128, 128, 128)

            # add data aug code...
            # patient_image, patient_label = random_mirror_flip(patient_image, patient_label)
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)


def get_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    # base_folder = pathlib.Path(get_brats_folder(on)).resolve()
    # print(base_folder)
    # assert base_folder.exists()
    # patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    #
    # kfold = KFold(3, shuffle=True, random_state=seed)
    # splits = list(kfold.split(patients_dir))
    # train_idx, val_idx = splits[fold_number]
    # len_val = len(val_idx)
    # val_index = val_idx[: len_val//2]
    # test_index = val_idx[len_val // 2 :]
    #
    # train = [patients_dir[i] for i in train_idx]
    # val = [patients_dir[i] for i in val_index]
    # test = [patients_dir[i] for i in test_index]

    train_folder = pathlib.Path(get_brats_folder("train")).resolve()
    valid_folder = pathlib.Path(get_brats_folder("val")).resolve()
    test_folder = pathlib.Path(get_brats_folder("test")).resolve()
    # print(base_folder)
    # assert base_folder.exists()
    train_dir = sorted([x for x in train_folder.iterdir() if x.is_dir()])
    valid_dir = sorted([x for x in valid_folder.iterdir() if x.is_dir()])
    test_dir = sorted([x for x in test_folder.iterdir() if x.is_dir()])

    # kfold = KFold(3, shuffle=True, random_state=seed)
    # splits = list(kfold.split(patients_dir))
    # train_idx, val_idx = splits[fold_number]
    # len_val = len(val_idx)
    # val_index = val_idx[: len_val // 2]
    # test_index = val_idx[len_val // 2:]

    train = train_dir
    val = valid_dir
    test = test_dir

    # return patients_dir
    train_dataset = Brats(train, training=True,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=False, data_aug=False,
                        normalisation=normalisation)
    bench_dataset = Brats(test, training=False, benchmarking=True,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset


def get_test_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    base_folder = pathlib.Path(get_test_brats_folder()).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    bench_dataset = Brats(patients_dir, training=False, benchmarking=True,
                          normalisation=normalisation, no_seg=True)
    return bench_dataset





