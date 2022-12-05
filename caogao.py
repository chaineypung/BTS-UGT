import SimpleITK as sitk
import numpy as np
import os
import glob

# big_file = r'E:\pcx\BraTS2020-correct'


# def connected_domain_1(image, index, area):
#     Image = image
#     image = Image[index, ...]
#     image_TC = Image[1, ...]
#     cca = sitk.ConnectedComponentImageFilter()
#     cca.SetFullyConnected(True)
#     _input = sitk.GetImageFromArray(image.astype(np.uint8))
#     output_ex = cca.Execute(_input)
#     stats = sitk.LabelShapeStatisticsImageFilter()
#     stats.Execute(output_ex)
#     num_label = cca.GetObjectCount()
#     num_list = [i for i in range(1, num_label+1)]
#     if num_list == []:
#         return Image
#     else:
#         area_list = []
#         sum_area = 0
#         for l in range(1, num_label + 1):
#             area_list.append(stats.GetNumberOfPixels(l))
#             sum_area += stats.GetNumberOfPixels(l)
#         if sum_area >= area:
#             final_label_list = num_list
#         else:
#             final_label_list = []
#         if final_label_list != []:
#             return Image
#         else:
#             output = sitk.GetArrayFromImage(output_ex)
#             for one_label in num_list:
#                 x, y, z, w, h, d = stats.GetBoundingBox(one_label)
#                 one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
#                 image_TC[z: z + d, y: y + h, x: x + w] = np.logical_or(image_TC[z: z + d, y: y + h, x: x + w], output[z: z + d, y: y + h, x: x + w])
#                 output[z: z + d, y: y + h, x: x + w] *= one_mask
#             mask = (output > 0).astype(np.bool)
#             image_TC = (image_TC > 0).astype(np.bool)
#             Image[index, ...] = mask
#             Image[1, ...] = image_TC
#             return Image
#
#
# files = os.listdir(big_file)
#
# for file in files:
#
#     files = os.listdir(big_file)
#
#     # Read data
#     ref_seg_img = sitk.ReadImage(big_file + '/' + file)
#     ref_seg = sitk.GetArrayFromImage(ref_seg_img)
#
#     # Split labels in different channels
#     refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
#     refmap_et = ref_seg == 4
#     refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
#     refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
#     refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
#
#     # Remove small points from ET
#     refmap = connected_domain_1(refmap, 0, 500)  # 300 is best
#
#     # Restore labels
#     et = refmap[0]
#     net = np.logical_and(refmap[1], np.logical_not(et))
#     ed = np.logical_and(refmap[2], np.logical_not(refmap[1]))
#     labelmap = np.zeros(refmap[0].shape)
#     labelmap[et] = 4
#     labelmap[net] = 1
#     labelmap[ed] = 2
#     labelmap = sitk.GetImageFromArray(labelmap)
#
#     # Add head information
#     labelmap.CopyInformation(ref_seg_img)
#
#     sitk.WriteImage(labelmap, big_file + '/' + file)
#     print(f"{file} done!")




# Wrong = sitk.ReadImage(r'E:\pcx\BraTS2020-correct\BraTS20_Validation_107.nii.gz')
# Correct = sitk.ReadImage(r'E:\pcx\BraTS2020-correct\other method\BraTS20_Validation_107.nii.gz')
#
# wrong = sitk.GetArrayFromImage(Wrong)
# correct = sitk.GetArrayFromImage(Correct)
#
# mask = (correct == 4)
# wrong[mask] = 4
#
# wrong = sitk.GetImageFromArray(wrong)
#
# wrong.CopyInformation(Wrong)
# sitk.WriteImage(wrong, r'E:\pcx\BraTS2020-correct\corrected\BraTS20_Validation_107.nii.gz')



big_file = r'C:\Users\zanchen\Desktop\submission'

files = os.listdir(big_file)

for file in files:


    # Read data
    ref_seg_img = sitk.ReadImage(big_file + '/' + file)
    sitk.WriteImage(ref_seg_img, big_file + '/' + file)
    print(f"{file} done!")


# big_file = 'E:/pcx/BraTS2020/MICCAI_BraTS2020_ValidationData'
#
# files = os.listdir(big_file)
#
# for file in files:
#
#
#     # Read data
#     ref_seg_img = sitk.ReadImage(big_file + '/' + file + '/' + file + '_seg.nii.gz')
#     sitk.WriteImage(ref_seg_img, 'C:/Users/zanchen/Desktop/submission' + '/' + file + '.nii.gz')
#     print(f"{file} done!")