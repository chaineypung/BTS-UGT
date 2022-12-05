import SimpleITK as sitk
import numpy as np
import nibabel as nib


def connected_domain_2(image):
    im = image
    image[image == 1] = 0
    image[image == 2] = 0
    image[image == 4] = 1
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label + 1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= 500:
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)
    NET = np.ones_like(output)
    for one_label in num_list:
        if one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        NET[z: z + d, y: y + h, x: x + w] *= one_mask
    NET = 1 - NET
    mask = (NET > 0)
    im[mask] = NET[mask]

    return im


Image = nib.load("E:\pcx\sample_dataset\Brats18_2013_1_1\Brats18_2013_0_1.nii.gz")
image = np.array(Image.dataobj)
output = connected_domain_2(image)