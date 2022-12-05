import SimpleITK as sitk

image_flair = sitk.ReadImage(r"E:\pcx\BraTS2021\train\BraTS2021_00000\BraTS2021_00000_flair.nii.gz")
image_t1 = sitk.ReadImage(r"E:\pcx\BraTS2021\train\BraTS2021_00000\BraTS2021_00000_t1.nii.gz")
image_t1ce = sitk.ReadImage(r"E:\pcx\BraTS2021\train\BraTS2021_00000\BraTS2021_00000_t1ce.nii.gz")
image_t2 = sitk.ReadImage(r"E:\pcx\BraTS2021\train\BraTS2021_00000\BraTS2021_00000_t2.nii.gz")

array_flair = sitk.GetArrayFromImage(image_flair)
array_t1 = sitk.GetArrayFromImage(image_t1)
array_t1ce = sitk.GetArrayFromImage(image_t1ce)
array_t2 = sitk.GetArrayFromImage(image_t2)

image = 3.0 * array_flair + 0.1 * array_t1 + 1.0 * array_t1ce + 1.0 * array_t2

image = sitk.GetImageFromArray(image)
image.CopyInformation(image_flair)
sitk.WriteImage(image, r"E:\pcx\BraTS2021\train\BraTS2021_00000\BraTS2021_00000_mix.nii.gz")